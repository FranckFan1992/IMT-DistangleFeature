import torch
from kornia.utils import create_meshgrid
import pdb
import numpy as np


@torch.no_grad()
    
def warp_kpts(
        kpts0,
        depth0,
        depth1,
        T_0to1,
        K0,
        K1,
        depth_thresh=0.2
):
    """
    Warp keypoints from image0 → image1 using depth + pose.

    Args:
        kpts0: [N, L, 2] pixel coords (x,y) in image0
        depth0: [N, H, W]
        depth1: [N, H, W]
        T_0to1: [N, 4, 4]
        K0: [N, 3, 3]
        K1: [N, 3, 3]

    Returns:
        valid_mask: [N, L]
        warped_kpts0: [N, L, 2]
    """

    N, H, W = depth0.shape
    device = depth0.device
    L = kpts0.shape[1]

    # -------------------------
    # 1. sample depth0
    # -------------------------

    kpts0_long = kpts0.round().long()

    kpts0_long[..., 0] = kpts0_long[..., 0].clamp(0, W - 1)
    kpts0_long[..., 1] = kpts0_long[..., 1].clamp(0, H - 1)

    kpts0_depth = torch.stack(
        [
            depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]]
            for i in range(N)
        ],
        dim=0,
    )

    depth_mask = kpts0_depth > 0
    
    # -------------------------
    # depth edge filtering
    # -------------------------

    # compute depth gradient
    grad_x = torch.abs(depth0[:, :, 1:] - depth0[:, :, :-1])
    grad_y = torch.abs(depth0[:, 1:, :] - depth0[:, :-1, :])

    # pad 回原大小
    grad_x = torch.nn.functional.pad(grad_x, (0,1,0,0))
    grad_y = torch.nn.functional.pad(grad_y, (0,0,0,1))

    depth_grad = grad_x + grad_y

    # sample gradient at keypoints
    grad_sampled = torch.stack(
        [
        depth_grad[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]]
        for i in range(N)
        ],
        dim=0,
    )

    # edge mask
    edge_mask = grad_sampled < 0.05

    # -------------------------
    # 2. backproject to camera
    # -------------------------

    ones = torch.ones_like(kpts0[..., :1])

    kpts0_h = torch.cat([kpts0, ones], dim=-1)
    kpts0_h = kpts0_h * kpts0_depth[..., None]

    K0_inv = torch.inverse(K0)

    kpts0_cam = torch.bmm(
        K0_inv,
        kpts0_h.transpose(1, 2)
    )

    # -------------------------
    # 3. transform to cam1
    # -------------------------

    R = T_0to1[:, :3, :3]
    t = T_0to1[:, :3, 3:4]

    kpts1_cam = torch.bmm(R, kpts0_cam) + t

    depth_computed = kpts1_cam[:, 2, :]

    # -------------------------
    # 4. project to image1
    # -------------------------

    kpts1_h = torch.bmm(K1, kpts1_cam).transpose(1, 2)

    warped_kpts0 = kpts1_h[:, :, :2] / (kpts1_h[:, :, 2:3] + 1e-8)

    # -------------------------
    # 5. covisible check
    # -------------------------

    x = warped_kpts0[..., 0]
    y = warped_kpts0[..., 1]

    covisible_mask = (
        (x > 0)
        & (x < W - 1)
        & (y > 0)
        & (y < H - 1)
    )

    # -------------------------
    # 6. sample depth1
    # -------------------------

    w_long = warped_kpts0.round().long()

    w_long[..., 0] = w_long[..., 0].clamp(0, W - 1)
    w_long[..., 1] = w_long[..., 1].clamp(0, H - 1)

    depth1_sampled = torch.stack(
        [
            depth1[i, w_long[i, :, 1], w_long[i, :, 0]]
            for i in range(N)
        ],
        dim=0,
    )

    depth1_mask = depth1_sampled > 0

    # -------------------------
    # 7. depth consistency
    # -------------------------

    depth_error = torch.abs(
        depth1_sampled - depth_computed
    ) / (depth1_sampled + 1e-8)

    consistent_mask = depth_error < depth_thresh

    # -------------------------
    # final mask
    # -------------------------

    valid_mask = (
        depth_mask
        & covisible_mask
        & depth1_mask
        & consistent_mask
        & edge_mask
    )

    return valid_mask, warped_kpts0
    
    


@torch.no_grad()
def spvs_coarse(data0, data1, scale = 4):
    """
        Supervise corresp with dense depth & camera poses
    """

    # 1. misc
    device = data0['img'].device
    N, _, H0, W0 = data0['img'].shape
    _, _, H1, W1 = data1['img'].shape
    
    #scale = 4
    scale0 = scale
    scale1 = scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    
    
    # -------- poses (numpy) --------
    T0 = data0['pose']      # (4,4)
    T1 = data1['pose']      # (4,4)

    T_0to1 = T1 @ torch.linalg.inv(T0)
    T_1to0 = torch.linalg.inv(T_0to1)
    

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt1_i = scale1 * grid_pt1_c

    # warp kpts bi-directionally and check reproj error
    # forward warp
    nonzero_m1, w_pt1_i  = warp_kpts(
        grid_pt1_i,
        data1['depth'],
        data0['depth'],
        T_1to0,
        data1['K'],
        data0['K']
    )

    # backward warp
    nonzero_m2, w_pt1_og = warp_kpts(
        w_pt1_i,
        data0['depth'],
        data1['depth'],
        T_0to1,
        data0['K'],
        data1['K']
    )

    # forward-backward reprojection error
    dist = torch.linalg.norm(grid_pt1_i - w_pt1_og, dim=-1)

    fb_thresh = 0.5

    mask_mutual = (
        (dist < fb_thresh)
        & nonzero_m1
        & nonzero_m2
    )

    # covisible check
    x = w_pt1_i[...,0]
    y = w_pt1_i[...,1]

    covisible = (
        (x > 0) &
        (x < W0-1) &
        (y > 0) &
        (y < H0-1)
    )

    mask_mutual = mask_mutual & covisible
    #_, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    # 构建GT 对应点 
    """
    batched_corrs = [
       [x0,y0,x1,y1], ...
       ]
    """
    
    batched_corrs = [ torch.cat([w_pt1_i[i, mask_mutual[i]] / scale0,
                       grid_pt1_i[i, mask_mutual[i]] / scale1],dim=-1) for i in range(len(mask_mutual))]
    
    
    
    
    # batched_corrs[i] 形状 = (Ni, 4) 每一行: [x0, y0, x1, y1] 图0上的点 (x0,y0) 对应 图1上的点 (x1,y1)


    #Remove repeated correspondences - this is important for network convergence
    # 去重的原因是，很多在相机空间的3D 点，投影到图像平面后，投影点的位置很近，再进行离散的网格处理后，可能会出现不同的3D点投影到
    # 同一位置
    corrs = []
    for pts in batched_corrs:
        lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = torch.float32) * -1  # 每个 coarse 像素位置存一个 [x0,y0,x1,y1] 初始化存-1 表示空
        lut_mat21 = torch.clone(lut_mat12)
        src_pts = pts[:, :2] / scale
        tgt_pts = pts[:, 2:] / scale
        try:
            # 如果有多个3D 点同时映射到同一个像素位置，自动覆盖从而实现去重的目的
            # 这里进行了两次去重，一次是src to tgt ，避免one to many 
            # 第二次是去重是tgt to src 避免many to one
            lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)  
            # 例如有A: (10,20) → (30,40) B: (10,20) → (35,45) 第二次赋值会：覆盖第一次 最终的结果是，一个source 像素只会保留一个匹配
            mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
            points = lut_mat12[mask_valid12]

            #Target-src check 
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            lut_mat21[tgt_pts[:,1].long(), tgt_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            # 例如有A: (10,20) → (30,40) B: (11,20) → (30,40) 第二次赋值会：覆盖第一次 最终的结果是，一个source 像素只会保留一个匹配
            mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
            points = lut_mat21[mask_valid21]

            corrs.append(points)
        except:
            pdb.set_trace()
            print('..')

    #Plot for debug purposes    
    # for i in range(len(corrs)):
    #     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2]*8, corrs[i][:, 2:]*8)


    return corrs


def spvs_coarse_orig_res(data0, data1, scale=4):
    """
    Supervise correspondences with dense depth & camera poses.
    Returns correspondences in ORIGINAL image resolution.
    Safe version: avoids pdb, handles empty points and out-of-bounds indices.

    Args:
        data0, data1: dict with keys 'img', 'depth', 'pose', 'K'
        scale: coarse downsampling factor

    Returns:
        corrs_orig_res: list of [Ni,4] tensors, each row [x0,y0,x1,y1] in original resolution
    """
    device = data0['img'].device
    N, _, H0, W0 = data0['img'].shape
    _, _, H1, W1 = data1['img'].shape

    scale0 = scale
    scale1 = scale

    h1, w1 = H1 // scale1, W1 // scale1

    # ---------- poses ----------
    T0 = data0['pose']  # (4,4)
    T1 = data1['pose']  # (4,4)
    T_0to1 = T1 @ torch.linalg.inv(T0)
    T_1to0 = torch.linalg.inv(T_0to1)

    # ---------- warp grids ----------
    # create coarse grid in downsampled resolution
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = grid_pt1_c * scale1  # map to original resolution

    # warp keypoints
    nonzero_m1, w_pt1_i  = warp_kpts(grid_pt1_i, data1['depth'], data0['depth'], T_1to0, data1['K'], data0['K'])
    nonzero_m2, w_pt1_og = warp_kpts(w_pt1_i, data0['depth'], data1['depth'], T_0to1, data0['K'], data1['K'])

    # 双向一致性
    dist = torch.linalg.norm(grid_pt1_i - w_pt1_og, dim=-1)
    mask_mutual = (dist < 1.5) & nonzero_m1 & nonzero_m2

    # 构建GT 对应点 (原图分辨率)
    batched_corrs = [
        torch.cat([w_pt1_i[i, mask_mutual[i]],
                   grid_pt1_i[i, mask_mutual[i]]], dim=-1) 
        for i in range(len(mask_mutual))
    ]  # [x0,y0,x1,y1] in ORIGINAL res

    # 去重逻辑，安全版
    corrs_orig_res = []
    for pts in batched_corrs:
        if pts.shape[0] == 0:
            # 没有匹配点，返回空 tensor
            corrs_orig_res.append(torch.zeros((0,4), device=device, dtype=torch.float32))
            continue

        # clamp 防止索引越界
        src_pts = pts[:, :2].clone()
        tgt_pts = pts[:, 2:].clone()
        src_pts[:,0] = src_pts[:,0].clamp(0, W1-1)
        src_pts[:,1] = src_pts[:,1].clamp(0, H1-1)
        tgt_pts[:,0] = tgt_pts[:,0].clamp(0, W1-1)
        tgt_pts[:,1] = tgt_pts[:,1].clamp(0, H1-1)

        lut_mat12 = torch.ones((H1, W1, 4), device=device, dtype=torch.float32) * -1
        lut_mat21 = torch.ones((H1, W1, 4), device=device, dtype=torch.float32) * -1

        # src -> tgt 去重
        lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
        mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
        points = lut_mat12[mask_valid12]

        # tgt -> src 去重
        if points.shape[0] > 0:
            src_pts2, tgt_pts2 = points[:, :2], points[:, 2:]
            lut_mat21[tgt_pts2[:,1].long(), tgt_pts2[:,0].long()] = torch.cat([src_pts2, tgt_pts2], dim=1)
            mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
            points = lut_mat21[mask_valid21]

        corrs_orig_res.append(points)

    return corrs_orig_res


import torch

def sample_fixed_points(corrs_orig_res, max_points=2000):
    """
    从原图匹配点中随机采样固定数量的点

    Args:
        corrs_orig_res: list of [Ni,4] tensors, 每行 [x0,y0,x1,y1] 
        max_points: 每张图最大匹配点数量

    Returns:
        corrs_sampled: list of [min(Ni,max_points),4] tensors
    """
    corrs_sampled = []

    for pts in corrs_orig_res:
        N = pts.shape[0]
        if N == 0:
            # 没有匹配点
            corrs_sampled.append(torch.zeros((0,4), device=pts.device))
            continue
        if N <= max_points:
            # 匹配点少于最大值，保留全部
            corrs_sampled.append(pts)
        else:
            # 随机采样 max_points 个点
            idx = torch.randperm(N, device=pts.device)[:max_points]
            corrs_sampled.append(pts[idx])

    return corrs_sampled