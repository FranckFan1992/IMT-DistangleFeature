import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

from methods.Xfeat.xfeat import *

def visualize_xfeat_matches(img1, img2, mkpts1, mkpts2):
    """
    img1, img2: numpy (H,W,3)
    mkpts1, mkpts2: (N,2)
    """

    # 👉 padding（和你之前一致）
    def pad_to_same_height(img1, img2):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        max_h = max(h1, h2)

        def pad(img, target_h):
            h, w, c = img.shape
            pad_h = target_h - h
            return np.pad(img, ((0,pad_h),(0,0),(0,0)), mode='constant')

        return pad(img1, max_h), pad(img2, max_h)

    img1, img2 = pad_to_same_height(img1, img2)

    concat_img = np.concatenate([img1, img2], axis=1)

    plt.figure(figsize=(15,7))
    plt.imshow(concat_img)

    for i in range(len(mkpts1)):
        pt1 = mkpts1[i]
        pt2 = mkpts2[i] + np.array([img1.shape[1], 0])

        plt.plot([pt1[0], pt2[0]],
                 [pt1[1], pt2[1]],
                 'y', linewidth=1)

    plt.title(f"XFeat Matches: {len(mkpts1)}")
    plt.axis('off')
    plt.show()

def get_xfeat_heatmap(xfeat, img_tensor):
    """
    返回 heatmap (numpy)
    """
    with torch.no_grad():
        x, _, _ = xfeat.preprocess_tensor(img_tensor)
        _, K1, _ = xfeat.net(x)

        heatmap = xfeat.get_kpts_heatmap(K1)  # (1,1,H,W)

    return heatmap.squeeze().cpu().numpy()

def visualize_xfeat_heatmap(img1, img2, hmap1, hmap2):
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    axes[0].imshow(img1)
    axes[0].imshow(hmap1, cmap='jet', alpha=0.5)
    axes[0].set_title("XFeat Heatmap img1")
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].imshow(hmap2, cmap='jet', alpha=0.5)
    axes[1].set_title("XFeat Heatmap img2")
    axes[1].axis('off')

    plt.show()
    
def load_image(path, device):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0).to(device)
    return img_tensor, img

# 初始化

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xfeat = XFeat(top_k = 50).to(device)

img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/307037213_48891bca3e_o.jpg"

#img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/511190120_77bee89b37_o.jpg"

#img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/frame-000413.color.png"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/frame-000240.color.png"

#img_path1 = "datasets/MegaDepth_v1/0022/dense0/testimgs/tabor1.png"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/testimgs/tabor2.jpg"

# 读取图像
img_tensor1, img1 = load_image(img_path1, device)
img_tensor2, img2 = load_image(img_path2, device)

# =========================
# 1️⃣ 匹配（直接用官方）
# =========================
mkpts1, mkpts2 = xfeat.match_xfeat(img1, img2)

# =========================
# 2️⃣ 可视化匹配
# =========================
visualize_xfeat_matches(img1, img2, mkpts1, mkpts2)

# =========================
# 3️⃣ heatmap
# =========================
hmap1 = get_xfeat_heatmap(xfeat, img_tensor1)
hmap2 = get_xfeat_heatmap(xfeat, img_tensor2)

visualize_xfeat_heatmap(img1, img2, hmap1, hmap2)
