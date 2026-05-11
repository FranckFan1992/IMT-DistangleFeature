import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from kornia.feature import R2D2
except ImportError as exc:
    raise ImportError(
        "kornia is required for matcher_r2d2.py. "
        "Install it with `pip install kornia` and make sure the version includes R2D2."
    ) from exc


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


def match_descriptors(desc1, desc2):
    desc1 = F.normalize(desc1, dim=-1)
    desc2 = F.normalize(desc2, dim=-1)
    sim = desc1 @ desc2.transpose(0, 1)
    idx12 = torch.argmax(sim, dim=1)
    idx21 = torch.argmax(sim, dim=0)
    matches = [(i.item(), idx12[i].item()) for i in range(idx12.shape[0]) if idx21[idx12[i]] == i]
    return matches


def visualize_matches(img1, img2, kpts1, kpts2, matches):
    img1_pad, img2_pad = pad_to_same_height(img1, img2)
    canvas = np.concatenate([img1_pad, img2_pad], axis=1)

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    for i, j in matches:
        pt1 = kpts1[i]
        pt2 = kpts2[j] + np.array([img1_pad.shape[1], 0])
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'y', linewidth=1)
    plt.scatter(kpts1[:, 0], kpts1[:, 1], s=10, c='cyan')
    plt.scatter(kpts2[:, 0] + img1_pad.shape[1], kpts2[:, 1], s=10, c='magenta')
    plt.axis('off')
    plt.title(f"R2D2 Matches: {len(matches)}")
    plt.show()


def visualize_keypoints(img, kpts, title_prefix=""):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.scatter(kpts[:, 0], kpts[:, 1], s=12, c='lime')
    plt.title(f"{title_prefix} Keypoints")
    plt.axis('off')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = R2D2(pretrained=True).to(device).eval()

    img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
    img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/307037213_48891bca3e_o.jpg"

    img_tensor1, img1 = load_image(img_path1, device)
    img_tensor2, img2 = load_image(img_path2, device)

    with torch.no_grad():
        out1 = model(img_tensor1)
        out2 = model(img_tensor2)

    # Kornia R2D2 may return tuple or dict depending on version.
    if isinstance(out1, tuple) and len(out1) >= 3:
        kpts1, desc1, scores1 = out1[0], out1[1], out1[2]
        kpts2, desc2, scores2 = out2[0], out2[1], out2[2]
    elif isinstance(out1, dict):
        kpts1, desc1, scores1 = out1['keypoints'], out1['descriptors'], out1.get('scores', None)
        kpts2, desc2, scores2 = out2['keypoints'], out2['descriptors'], out2.get('scores', None)
    else:
        raise RuntimeError('Unsupported R2D2 output format. Please check Kornia version.')

    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.squeeze(0).cpu().numpy()
    if isinstance(kpts2, torch.Tensor):
        kpts2 = kpts2.squeeze(0).cpu().numpy()

    matches = match_descriptors(desc1.squeeze(0), desc2.squeeze(0))
    print(f"R2D2 keypoints: image1={len(kpts1)}, image2={len(kpts2)}, matches={len(matches)}")

    visualize_matches(img1, img2, kpts1, kpts2, matches)
    visualize_keypoints(img1, kpts1, title_prefix="Image1")
    visualize_keypoints(img2, kpts2, title_prefix="Image2")


if __name__ == '__main__':
    main()
