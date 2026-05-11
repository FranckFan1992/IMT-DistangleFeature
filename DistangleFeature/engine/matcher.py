import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from methods.EmbPose.varkpnetmodel import VUDNet


def pad_to_same_height(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    max_h = max(h1, h2)

    def pad(img, target_h):
        h, w, c = img.shape
        pad_h = target_h - h
        return np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode='constant')

    return pad(img1, max_h), pad(img2, max_h)


def load_image(path, device):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor, img


def extract_keypoints(heatmap, reliability, num_keypoints=200, border=16, min_score=1e-4):
    score = heatmap * reliability
    score = score.squeeze(0).squeeze(0)

    max_pool = F.max_pool2d(score.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    peak_mask = score == max_pool.squeeze(0).squeeze(0)
    score = score * peak_mask.float()

    if border > 0:
        score[:border, :] = 0
        score[-border:, :] = 0
        score[:, :border] = 0
        score[:, -border:] = 0

    flat = score.view(-1)
    topk = min(num_keypoints, flat.numel())
    values, idx = torch.topk(flat, k=topk)
    keep = values > min_score
    values = values[keep]
    idx = idx[keep]

    if idx.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), score

    ys = (idx // score.size(1)).cpu().numpy()
    xs = (idx % score.size(1)).cpu().numpy()
    coords_feat = np.stack([xs, ys], axis=-1).astype(np.float32)
    scores = values.cpu().numpy().astype(np.float32)
    return coords_feat, scores, score


def map_feat_coords_to_image(coords_feat, img_shape, feat_shape):
    H, W = img_shape[:2]
    Hf, Wf = feat_shape
    scale_x = W / Wf
    scale_y = H / Hf
    coords_img = coords_feat.copy()
    coords_img[:, 0] *= scale_x
    coords_img[:, 1] *= scale_y
    return coords_img


def coords_to_feat(feat_map, coords_feat):
    B, C, H, W = feat_map.shape
    if coords_feat.shape[0] == 0:
        return np.zeros((0, C), dtype=np.float32)
    x = np.clip(np.round(coords_feat[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(coords_feat[:, 1]).astype(int), 0, H - 1)
    feat = feat_map[0, :, y, x].permute(1, 0)
    feat = F.normalize(feat, dim=1)
    return feat.cpu().numpy()


def visualize_matches(img1, img2, coords1, coords2, matches):
    img1_pad, img2_pad = pad_to_same_height(img1, img2)
    concat_img = np.concatenate([img1_pad, img2_pad], axis=1)

    plt.figure(figsize=(16, 8))
    plt.imshow(concat_img)
    for i, j in matches:
        pt1 = coords1[i]
        pt2 = coords2[j] + np.array([img1_pad.shape[1], 0])
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'y', linewidth=1)
    plt.axis('off')
    plt.title(f"Matches: {len(matches)}")
    plt.show()


def visualize_single_maps(img, heatmap, reliability, inv_map, geo_map, app_map, title_prefix=""):
    img_np = img if isinstance(img, np.ndarray) else img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    heatmap = heatmap.squeeze().cpu().numpy()
    reliability = reliability.squeeze().cpu().numpy()
    inv_map = inv_map.squeeze().cpu().numpy()
    geo_map = geo_map.squeeze().cpu().numpy()
    app_map = app_map.squeeze().cpu().numpy()

    if inv_map.ndim == 3:
        inv_vis = inv_map.mean(axis=0)
    else:
        inv_vis = inv_map
    if geo_map.ndim == 3:
        geo_vis = geo_map.mean(axis=0)
    else:
        geo_vis = geo_map
    if app_map.ndim == 3:
        app_vis = app_map.mean(axis=0)
    else:
        app_vis = app_map

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"{title_prefix} Image")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(heatmap, cmap='hot')
    axes[0, 1].set_title(f"{title_prefix} Heatmap")
    axes[0, 1].axis('off')
    axes[0, 2].imshow(reliability, cmap='viridis')
    axes[0, 2].set_title(f"{title_prefix} Reliability")
    axes[0, 2].axis('off')
    axes[1, 0].imshow(inv_vis, cmap='plasma')
    axes[1, 0].set_title(f"{title_prefix} Inv Map")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(geo_vis, cmap='magma')
    axes[1, 1].set_title(f"{title_prefix} Geo Map")
    axes[1, 1].axis('off')
    axes[1, 2].imshow(app_vis, cmap='cividis')
    axes[1, 2].set_title(f"{title_prefix} App Map")
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.show()


def visualize_all_maps(img, out, title_prefix=""):
    inv_map = out.get('f_inv', out.get('f_noise', out.get('f_app', None)))
    geo_map = out.get('f_geo', inv_map)
    app_map = out.get('f_app', out.get('f_noise', inv_map))
    visualize_single_maps(img, out['heatmap'], out['reliability'], inv_map, geo_map, app_map, title_prefix)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = VUDNet(feature_dim=128, dim_geo=32, dim_noise=16, pose_dim=9, pose_embed=128)
    net = net.to(device)

    checkpoint_path = "checkpoints/kpnet_iter_xfeat_45000.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    net.load_state_dict(checkpoint, strict=False)
    net.eval()

    #img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
    #img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/511190120_77bee89b37_o.jpg"
    
    img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
    img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/307037213_48891bca3e_o.jpg"

    img_tensor1, img1 = load_image(img_path1, device)
    img_tensor2, img2 = load_image(img_path2, device)

    with torch.no_grad():
        out1 = net(img_tensor1)
        out2 = net(img_tensor2)

    coords_feat1, scores1, _ = extract_keypoints(out1['heatmap'], out1['reliability'], num_keypoints=50, border=16)
    coords_feat2, scores2, _ = extract_keypoints(out2['heatmap'], out2['reliability'], num_keypoints=50, border=16)

    coords1 = map_feat_coords_to_image(coords_feat1, img1.shape, out1['heatmap'].shape[-2:])
    coords2 = map_feat_coords_to_image(coords_feat2, img2.shape, out2['heatmap'].shape[-2:])

    print(f"Image1 keypoints: {len(coords1)}, Image2 keypoints: {len(coords2)}")

    f1 = coords_to_feat(out1['f_inv'], coords_feat1)
    f2 = coords_to_feat(out2['f_inv'], coords_feat2)

    sim = f1 @ f2.T
    idx12 = np.argmax(sim, axis=1)
    idx21 = np.argmax(sim, axis=0)
    matches = [(i, j) for i, j in enumerate(idx12) if idx21[j] == i]
    print(f"Total MNN matches: {len(matches)}")

    visualize_matches(img1, img2, coords1, coords2, matches)
    visualize_all_maps(img1, out1, title_prefix="Image1")
    visualize_all_maps(img2, out2, title_prefix="Image2")



if __name__ == '__main__':
    main()