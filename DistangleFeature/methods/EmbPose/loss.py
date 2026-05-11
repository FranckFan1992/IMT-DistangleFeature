import torch
import torch.nn.functional as F


def _zero_loss(device, requires_grad=True):
    return torch.tensor(0.0, device=device, requires_grad=requires_grad)


def mv_infonce_masked(f_inv, visibility, tau=0.07):
    """
    f_inv: [N, V, C]
    visibility: [N, V] (bool)
    """
    _, V, _ = f_inv.shape
    f = F.normalize(f_inv, dim=-1)

    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]
            if mask.sum() < 2:
                continue

            fi = f[mask, i]
            fj = f[mask, j]
            sim = fi @ fj.t() / tau
            labels = torch.arange(sim.shape[0], device=sim.device)

            loss = loss + 0.5 * (
                F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.t(), labels)
            )
            count += 1

    if count == 0:
        return _zero_loss(f.device)
    return loss / count


def invariant_consistency_loss(f_inv, visibility):
    """
    Pull invariant descriptors of the same 3D point together across views.
    """
    V = f_inv.shape[1]
    loss = 0.0
    count = 0

    f_inv = F.normalize(f_inv, dim=-1)
    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]
            if mask.sum() == 0:
                continue
            loss = loss + F.mse_loss(f_inv[mask, i], f_inv[mask, j])
            count += 1

    if count == 0:
        return _zero_loss(f_inv.device)
    return loss / count


def reconstruction_loss(kpnet, f_inv, f_geo, f_noise, shared, visibility):
    """
    Reconstruct each view's backbone descriptor from the three disentangled parts.
    """
    _, V, _ = f_inv.shape
    loss = 0.0
    count = 0

    for v in range(V):
        mask = visibility[:, v]
        if mask.sum() == 0:
            continue
        pred = kpnet.reconstruct_feature(
            f_inv[mask, v],
            f_geo[mask, v],
            f_noise[mask, v]
        )
        loss = loss + F.mse_loss(pred, shared[mask, v])
        count += 1

    if count == 0:
        return _zero_loss(f_inv.device)
    return loss / count


def geometry_consistency_loss(kpnet, f_geo, poses, visibility):
    """
    Predict target-view geometry descriptor from source-view geometry + relative pose.
    """
    _, V, _ = f_geo.shape
    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            mask = visibility[:, i] & visibility[:, j]
            if mask.sum() == 0:
                continue

            pred_geo_j = kpnet.transform_geo(
                f_geo[mask, i],
                poses[i],
                poses[j]
            )
            loss = loss + F.mse_loss(pred_geo_j, f_geo[mask, j])
            count += 1

    if count == 0:
        return _zero_loss(f_geo.device)
    return loss / count


def cross_view_reconstruction_loss(kpnet, f_inv, f_geo, f_noise, shared, poses, visibility):
    """
    Use source invariant descriptor + pose-transferred geometry + target noise
    to reconstruct the target-view backbone descriptor.
    """
    _, V, _ = f_inv.shape
    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            mask = visibility[:, i] & visibility[:, j]
            if mask.sum() == 0:
                continue

            pred_shared_j, _ = kpnet.predict_view(
                f_inv[mask, i],
                f_geo[mask, i],
                f_noise[mask, j],
                poses[i],
                poses[j]
            )
            loss = loss + F.mse_loss(pred_shared_j, shared[mask, j])
            count += 1

    if count == 0:
        return _zero_loss(f_inv.device)
    return loss / count


def orthogonality_loss(*features):
    """
    Encourage disentangled branches to be decorrelated.
    """
    if len(features) < 2:
        return _zero_loss(features[0].device if len(features) == 1 else "cpu", requires_grad=False)

    loss = 0.0
    count = 0
    device = features[0].device

    processed = []
    for feat in features:
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat = F.normalize(feat, dim=0)
        processed.append(feat)

    for i in range(len(processed)):
        for j in range(i + 1, len(processed)):
            cross_corr = processed[i].transpose(0, 1) @ processed[j]
            cross_corr = cross_corr / max(processed[i].shape[0], 1)
            loss = loss + (cross_corr ** 2).mean()
            count += 1

    if count == 0:
        return _zero_loss(device)
    return loss / count


def noise_regularization_loss(f_noise):
    """
    Keep the residual branch from trivially absorbing all information.
    """
    return f_noise.pow(2).mean()


def heatmap_mse_loss(pred, target):
    return F.mse_loss(pred, target)


def heatmap_topk_loss(pred, target, topk=1024):
    """
    pred: [B, 1, H, W]
    target: [B, 1, H, W]
    """
    B = pred.shape[0]
    loss = pred.new_tensor(0.0)

    for b in range(B):
        p = pred[b].reshape(-1)
        t = target[b].reshape(-1)

        if t.sum() < 1e-6:
            continue

        k = min(topk, t.numel())
        idx = torch.topk(t, k=k).indices
        loss = loss + (-torch.log(p[idx] + 1e-6).mean())

    return loss / max(B, 1)


def heatmap_nms_loss(pred):
    """
    Encourage local maxima.
    """
    maxpool = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
    return F.l1_loss(pred, maxpool)


def heatmap_loss(pred, target, topk=1024):
    loss_mse = heatmap_mse_loss(pred, target)
    loss_topk = heatmap_topk_loss(pred, target, topk=topk)
    loss_nms = heatmap_nms_loss(pred)
    return loss_mse + 0.5 * loss_topk + 0.1 * loss_nms


def reliability_loss_from_confidence(
    rel_pred,
    f_inv,
    visibility,
    topk=128,
    eps=1e-6,
):
    """
    rel_pred: [N, V, 1]
    f_inv: [N, V, C]
    visibility: [N, V]
    """
    N, V, _ = f_inv.shape
    f = F.normalize(f_inv, dim=2)

    conf = torch.zeros(N, V, device=f.device)
    count = 0

    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 10:
                continue

            fi = f[idx, i]
            fj = f[idx, j]
            sim = fi @ fj.t()

            k_eff = min(topk, sim.size(1))
            topk_val, topk_idx = torch.topk(sim, k=k_eff, dim=1)
            p_ij = F.softmax(topk_val, dim=1)

            sim_t = sim.t()
            topk_val_t = torch.gather(sim_t, 1, topk_idx.t()).t()
            p_ji = F.softmax(topk_val_t, dim=1)

            conf_ij = (p_ij * p_ji).sum(dim=1)
            conf[idx, i] += conf_ij
            conf[idx, j] += conf_ij
            count += 1

    conf = conf / (count + eps)
    conf = conf.clamp(eps, 1.0)
    target = conf.detach()

    rel_pred = rel_pred.squeeze(-1)
    loss = F.mse_loss(rel_pred, target)
    return loss, target


def total_disentangle_loss(
    kpnet,
    f_inv,
    f_geo,
    f_noise,
    shared,
    poses,
    visibility,
    w_inv_nce=1.0,
    w_inv_consistency=1.0,
    w_recon=1.0,
    w_geo=1.0,
    w_cross_recon=0.5,
    w_ortho=0.1,
    w_noise_reg=1e-3,
):
    """
    Point-wise multi-view disentanglement objective.
    """
    loss_inv_nce = mv_infonce_masked(f_inv, visibility)
    loss_inv_consistency = invariant_consistency_loss(f_inv, visibility)
    loss_recon = reconstruction_loss(kpnet, f_inv, f_geo, f_noise, shared, visibility)
    loss_geo = geometry_consistency_loss(kpnet, f_geo, poses, visibility)
    loss_cross_recon = cross_view_reconstruction_loss(
        kpnet, f_inv, f_geo, f_noise, shared, poses, visibility
    )

    visible_mask = visibility.unsqueeze(-1)
    f_inv_valid = f_inv[visible_mask.expand_as(f_inv)].view(-1, f_inv.shape[-1])
    f_geo_valid = f_geo[visible_mask.expand_as(f_geo)].view(-1, f_geo.shape[-1])
    f_noise_valid = f_noise[visible_mask.expand_as(f_noise)].view(-1, f_noise.shape[-1])

    if f_inv_valid.shape[0] == 0:
        loss_ortho = _zero_loss(f_inv.device)
        loss_noise_reg = _zero_loss(f_inv.device)
    else:
        loss_ortho = orthogonality_loss(f_inv_valid, f_geo_valid, f_noise_valid)
        loss_noise_reg = noise_regularization_loss(f_noise_valid)

    total = (
        w_inv_nce * loss_inv_nce +
        w_inv_consistency * loss_inv_consistency +
        w_recon * loss_recon +
        w_geo * loss_geo +
        w_cross_recon * loss_cross_recon +
        w_ortho * loss_ortho +
        w_noise_reg * loss_noise_reg
    )

    return {
        "total": total,
        "inv_nce": loss_inv_nce,
        "inv_consistency": loss_inv_consistency,
        "reconstruction": loss_recon,
        "geo": loss_geo,
        "cross_reconstruction": loss_cross_recon,
        "orthogonality": loss_ortho,
        "noise_regularization": loss_noise_reg,
    }
