import torch
import torch.nn.functional as F

def extract_keypoints(heatmap, topk=1024):
    """
    heatmap: [1,1,H,W]
    """

    B, _, H, W = heatmap.shape

    # NMS
    maxpool = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    keep = (heatmap == maxpool).float()
    heatmap = heatmap * keep

    # flatten
    scores = heatmap.view(-1)

    k = min(topk, scores.numel())
    vals, idx = torch.topk(scores, k)

    ys = idx // W
    xs = idx % W

    keypoints = torch.stack([xs, ys], dim=1).float()

    return keypoints, vals

def sample_descriptors(fmap, keypoints):
    """
    fmap: [1,C,H,W]
    keypoints: [K,2]
    """

    K = keypoints.shape[0]

    xs = keypoints[:, 0] / (fmap.shape[3] - 1) * 2 - 1
    ys = keypoints[:, 1] / (fmap.shape[2] - 1) * 2 - 1

    grid = torch.stack([xs, ys], dim=1).view(1, K, 1, 2)

    desc = F.grid_sample(fmap, grid, align_corners=True)
    desc = desc.view(fmap.shape[1], K).t()

    return F.normalize(desc, dim=1)

def inference(model, image, topk=1024):

    model.eval()

    with torch.no_grad():
        out = model(image)

        heatmap = out["heatmap"]
        f_inv = out["f_inv"]
        sigma = out["sigma"]
        reliability = out["reliability"]

        keypoints, scores = extract_keypoints(heatmap, topk)

        desc = sample_descriptors(f_inv, keypoints)

    return {
        "keypoints": keypoints,
        "descriptors": desc,
        "scores": scores
    }


"""
final_score = heatmap * reliability * (1 - sigma)
"""