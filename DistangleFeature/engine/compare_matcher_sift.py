import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 可视化匹配（保持你的逻辑）
# =========================
def visualize_matches(img1, img2, mkpts1, mkpts2):

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

    plt.title(f"SIFT Matches: {len(mkpts1)}")
    plt.axis('off')
    plt.show()


# =========================
# 读取图像
# =========================
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# =========================
# SIFT 匹配
# =========================
def sift_match(img1, img2, top_k=200):

    # 转灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # 初始化 SIFT
    sift = cv2.SIFT_create(nfeatures=top_k, contrastThreshold=0.01)

    # 提取关键点 & 描述子
    kpts1, des1 = sift.detectAndCompute(gray1, None)
    kpts2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher + ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 转成 numpy 点
    mkpts1 = np.array([kpts1[m.queryIdx].pt for m in good_matches])
    mkpts2 = np.array([kpts2[m.trainIdx].pt for m in good_matches])

    return mkpts1, mkpts2, kpts1, kpts2


# =========================
# （可选）用响应值做“伪heatmap”
# =========================
def get_sift_heatmap(img, keypoints):

    h, w, _ = img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < w and 0 <= y < h:
            heatmap[y, x] = kp.response

    # 归一化
    heatmap = cv2.GaussianBlur(heatmap, (15,15), 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap


def visualize_heatmap(img1, img2, hmap1, hmap2):
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    axes[0].imshow(img1)
    axes[0].imshow(hmap1, cmap='jet', alpha=0.5)
    axes[0].set_title("SIFT Heatmap img1")
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].imshow(hmap2, cmap='jet', alpha=0.5)
    axes[1].set_title("SIFT Heatmap img2")
    axes[1].axis('off')

    plt.show()


# =========================
# 主程序
# =========================
#img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/frame-000413.color.png"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/frame-000240.color.png"
#img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/307037213_48891bca3e_o.jpg"

img_path1 = "datasets/MegaDepth_v1/0022/dense0/testimgs/tabor1.png"
img_path2 = "datasets/MegaDepth_v1/0022/dense0/testimgs/tabor2.jpg"



img1 = load_image(img_path1)
img2 = load_image(img_path2)

# 1️⃣ SIFT匹配
mkpts1, mkpts2, kpts1, kpts2 = sift_match(img1, img2, top_k=80)

# 2️⃣ 可视化匹配
visualize_matches(img1, img2, mkpts1, mkpts2)

# 3️⃣ heatmap（SIFT版）
hmap1 = get_sift_heatmap(img1, kpts1)
hmap2 = get_sift_heatmap(img2, kpts2)

visualize_heatmap(img1, img2, hmap1, hmap2)