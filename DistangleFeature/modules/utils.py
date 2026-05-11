
import torch 
import numpy as np
import math
import os.path as osp
import cv2
import torch.nn.functional as F
from skimage.io import imread
import pickle
# -----------------------------
# 计算相机中心
# -----------------------------
def compute_camera_center(qvec, tvec):
    """qvec: quaternion, tvec: translation"""
    # 先转为旋转矩阵
    w, x, y, z = qvec
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y- z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x**2 + y**2)]
    ])
    C = -R.T @ tvec
    return C

# -----------------------------
# 计算 viewing angle
# -----------------------------
def compute_viewing_angle(X, C, ref_C):
    v1 = X - C
    v2 = X - ref_C
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_theta)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getIntrinsic(FoVx, width, height):
    K = np.eye(3)
    focal_length = fov2focal(FoVx, width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    return K

def getExtrinsic(R, t):
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3,3] = t
    return pose


def load_depth_from_png(tiff_file_path):
    depth = cv2.imread(tiff_file_path, cv2.IMREAD_ANYDEPTH)
    depth[depth==65535]=0
    return depth



def load_raw_data(
    base_dir, scene_info, idx, read_img=True):
    
    
    pose = scene_info["pose_list"][idx]
    K = scene_info["intrinsics_list"][idx]
    
    file_name = scene_info["image_name_list"][idx]
    depth_file_name = scene_info["depth_name_list"][idx]

    img = None
    img_path = osp.join(base_dir, file_name)
    if not osp.isfile(img_path):
        print("ERROR: img_path does not exist:", img_path)

    depth_file_name = osp.join(base_dir, depth_file_name)
    if read_img:
        img = imread(img_path)
    depth = load_depth_from_png(depth_file_name)

    depth=depth.astype(np.float32)/1000
    depth[depth < 1e-5] = 0
    return img, depth, pose, K


def load_data(base_dir, scene_info, idx):

    img, depth, pose, K = load_raw_data(base_dir, scene_info, idx, read_img=True)

    img = img.astype(np.float32).transpose(2,0,1)
    
    return {
        "pose": pose.astype(np.float32),
        "K": K.astype(np.float32),
        "depth": depth.astype(np.float32),
        "img": img
    }



def rotation_matrix_to_quaternion(R):
    """
    R: (3,3)
    return: (4) quaternion (qx,qy,qz,qw)
    """

    m00 = R[0,0]
    m01 = R[0,1]
    m02 = R[0,2]
    m10 = R[1,0]
    m11 = R[1,1]
    m12 = R[1,2]
    m20 = R[2,0]
    m21 = R[2,1]
    m22 = R[2,2]

    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-8)) / 2
    qx = (m21 - m12) / (4 * qw)
    qy = (m02 - m20) / (4 * qw)
    qz = (m10 - m01) / (4 * qw)

    q = torch.stack([qx, qy, qz, qw], dim=0)

    return q


def pose_matrix_to_7d(pose):
    """
    pose: (4,4)
    return: (7)
    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    q = rotation_matrix_to_quaternion(R)

    q = F.normalize(q, dim=0)
    pose7 = torch.cat([q, t], dim=0)

    return pose7

def sample_map_at_coords(fmap, coords, H, W):
    """
    fmap:   [1, C, Hc, Wc]
    coords: [N, 2]  (原图坐标)
    H, W:   原图尺寸
    """

    device = fmap.device
    coords = coords.to(device).float()

    # ===== ⭐ 1. 先缩放到 feature map 尺度（加在这里） =====
    _, _, Hc, Wc = fmap.shape

    coords = coords.clone()
    coords[..., 0] = coords[..., 0] * (Wc / W)
    coords[..., 1] = coords[..., 1] * (Hc / H)

    # ===== 2. 再 normalize 到 [-1,1] =====
    coords_norm = coords.clone()
    coords_norm[..., 0] = coords_norm[..., 0] / (Wc - 1) * 2 - 1
    coords_norm[..., 1] = coords_norm[..., 1] / (Hc - 1) * 2 - 1

    # ===== 3. reshape =====
    coords_norm = coords_norm.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]

    # ===== 4. grid_sample =====
    sampled = F.grid_sample(
        fmap, coords_norm,
        mode='bilinear',
        align_corners=True
    )  # [1, C, N, 1]

    # ===== 5. reshape → [N,C] =====
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)

    return sampled


def check_accuracy(X, Y, pts1 = None, pts2 = None, plot=False):
    with torch.no_grad():
        #dist_mat = torch.cdist(X,Y)
        dist_mat = X @ Y.t()
        nn = torch.argmax(dist_mat, dim=1)
        #nn = torch.argmin(dist_mat, dim=1)
        correct = nn == torch.arange(len(X), device = X.device)

        if pts1 is not None and plot:
            import matplotlib.pyplot as plt
            canvas = torch.zeros((120, 160),device=X.device)
            pts1 = pts1[~correct]
            canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
            canvas = canvas.cpu().numpy()
            plt.imshow(canvas), plt.show()

        acc = correct.sum().item() / len(X)
        return acc
    
def save_pairs(train_pairs, test_pairs, save_path):

    data = {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print("Pairs saved to", save_path)
    
def load_pairs(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["train_pairs"], data["test_pairs"]
    


def parse_7scenes_image_name(name):
    seq_str, frame_str = name.split("/")
    seq_id = int(seq_str.split("-")[1])
    frame_id = int(frame_str.split("-")[1].split(".")[0])
    return seq_id, frame_id

def build_frame_index(images):
    frame_index = {}
    for img_id, img in images.items():
        seq_id, frame_id = parse_7scenes_image_name(img.name)
        frame_index[img_id] = {"seq": seq_id, "frame": frame_id}
    return frame_index