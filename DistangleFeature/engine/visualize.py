import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from methods.EmbPose.varkpnetmodel import *

# -------------------------
# NMS
# -------------------------
def nms_2d(score_map, nms_radius=4, top_k=500):
    score = torch.tensor(score_map).unsqueeze(0).unsqueeze(0)

    max_pool = F.max_pool2d(
        score,
        kernel_size=2*nms_radius+1,
        stride=1,
        padding=nms_radius
    )

    keep = (score == max_pool) & (score > 0)
    keep = keep.squeeze()

    ys, xs = torch.where(keep)
    scores = score_map[ys, xs]

    if len(scores) > top_k:
        idx = np.argsort(scores)[-top_k:]
        ys = ys[idx]
        xs = xs[idx]
        scores = scores[idx]

    keypoints = [(int(y), int(x), float(s)) for y, x, s in zip(ys, xs, scores)]
    return keypoints


# -------------------------
# 坐标 ×4
# -------------------------
def upscale_keypoints(kpts, scale=4):
    return [(y*scale, x*scale, s) for (y,x,s) in kpts]


# -------------------------
# 画点
# -------------------------
def draw_keypoints(img, keypoints, color):
    img_vis = img.copy()
    for (y, x, s) in keypoints:
        cv2.circle(img_vis, (int(x), int(y)), 5, color, -1)
    return img_vis


# -------------------------
# 主函数（🔥包含加载模型）
# -------------------------
def run_test(image_path, checkpoint_path, device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ===== 1. 初始化模型 =====
    model = VUDNet(feature_dim=64, dim_geo=32,
                 dim_app=16)
    
    # ===== 2. 加载权重 =====
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"✅ Loaded model from {checkpoint_path}")

    # ===== 3. 读图 =====
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0).to(device)

    # ===== 4. forward =====
    with torch.no_grad():
        out = model(img_tensor)

    # 👉 H/4, W/4
    var_map = out["sigma"][0,0].cpu().numpy()
    rel_map = out["reliability"][0,0].cpu().numpy()

    print("var range:", var_map.min(), var_map.max())
    print("rel range:", rel_map.min(), rel_map.max())

    # ===== 5. 构造 score =====
    # ⭐ 推荐：融合（最稳）
    #score_map = rel_map / (var_map + 1e-6)

    # ===== 6. NMS（低分辨率）=====
    #kpts = nms_2d(score_map, nms_radius=4, top_k=500)

    # ===== 7. 映射回原图 =====
    #kpts_up = upscale_keypoints(kpts, scale=4)

    # ===== 8. 可视化 =====
    #img_kpts = draw_keypoints(img, kpts_up, (0,255,0))

    # ===== 9. heatmap（可选）=====
    var_vis = cv2.resize(var_map, (W, H))
    rel_vis = cv2.resize(rel_map, (W, H))

    # ===== 10. 显示 =====
    plt.figure(figsize=(18,5))

    plt.subplot(1,4,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    #plt.subplot(1,4,2)
    #plt.title("Keypoints")
    #plt.imshow(img_kpts)
    #plt.axis('off')

    plt.subplot(1,4,3)
    plt.title("Variance")
    plt.imshow(var_vis, cmap='jet')
    plt.colorbar()

    plt.subplot(1,4,4)
    plt.title("Reliability")
    plt.imshow(rel_vis, cmap='jet')
    plt.colorbar()

    plt.show()

    #return kpts_up
    return None

if __name__ == "__main__":
    data_path = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
    #data_path = "datasets/MegaDepth_v1/0022/dense0/imgs/8232974_61eb861d2c_o.jpg"
    #data_path = "datasets/MegaDepth_v1/0022/dense0/testimgs/republique1.jpg"
    cpkt_save_path = "checkpoints/kpnet_iter_27999.pth"
    run_test(data_path, cpkt_save_path)