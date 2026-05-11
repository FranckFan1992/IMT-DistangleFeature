import torch
from kornia.utils import create_meshgrid
import numpy as np



# 也就是 当某个点出现在5个view 时，repeatability = 1  当只出现在 4 个views 是， repeatability = 0.8
# v = 3 repeatability = 0.6   v = 2 repeatability = 0.4
def build_heatmap_target(
    corrs_k,
    visibility,
    H, W,
    downsample=4,
    device="cuda"
):
    """
    corrs_k: [N, V, 2] (pixel coords in original image)
    visibility: [N, V]
    """
    


    N, V, _ = corrs_k.shape

    Hc = H // downsample
    Wc = W // downsample

    heatmap = torch.zeros((V, 1, Hc, Wc), device=device)
    count_map = torch.zeros_like(heatmap)

    for v in range(V):
        coords = corrs_k[:, v]  # [N,2]
        vis = visibility[:, v]
        
        coords = coords.to(device)
        vis = vis.to(device)

        coords = coords[vis]
        if coords.shape[0] == 0:
            continue

        # ↓ downsample
        xs = (coords[:, 0] / downsample).long()
        ys = (coords[:, 1] / downsample).long()

        xs = xs.clamp(0, Wc - 1)
        ys = ys.clamp(0, Hc - 1)

        # repeatability（=该 subset 的 view 数）
        repeatability = float(V) / 5.0   # normalize to [0,1]

        score = torch.full(
            (coords.shape[0],),
            repeatability,
            device=device,
            dtype=heatmap.dtype
        )  # [N_vis]

        """
        for i in range(xs.shape[0]):
            x, y = xs[i], ys[i]
            heatmap[v, 0, y, x] += score[i]
            count_map[v, 0, y, x] += 1
        """
        #####################################
        # ✅ vectorized（GPU并行）
        heatmap[v, 0].index_put_(
            (ys, xs),
            score,
            accumulate=True
        )

        count_map[v, 0].index_put_(
            (ys, xs),
            torch.ones_like(score),
            accumulate=True
        )
        ####################################

    heatmap = heatmap / (count_map + 1e-6)
    return heatmap


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long().clip(0, 2000-1)

    # 边界深度清0
    depth0[:, 0, :] = 0 ; depth1[:, 0, :] = 0 
    depth0[:, :, 0] = 0 ; depth1[:, :, 0] = 0 

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth > 0

    # Draw cross marks on the image for each keypoint
    # for b in range(len(kpts0)):
    #     fig, ax = plt.subplots(1,2)
    #     depth_np = depth0.numpy()[b]
    #     depth_np_plot = depth_np.copy()
    #     for x, y in kpts0_long[b, nonzero_mask[b], :].numpy():
    #         cv2.drawMarker(depth_np_plot, (x, y), (255), cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #     ax[0].imshow(depth_np)
    #     ax[1].imshow(depth_np_plot)

    # Unproject  将I0 上的点反投影到相机空间(这里只需要用到深度和内参)
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform  有了I0 到 I1 之间在相机空间的转换矩阵 T_0to1， 将I0 在相机空间的坐标住那换到I1 的相机空间下
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    #h, w = depth1.shape[1:3]
    #covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
    #     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0

    # w_kpts0_depth = torch.stack(
    #     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    # )  # (N, L)
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2


    valid_mask = nonzero_mask #* consistent_mask* covisible_mask 

    return valid_mask, w_kpts0
def generate_exclusive_subsets(batch_data, subset_views_list=[5,4,3,2], scale=4):
    """
    生成递减互斥的多视图匹配子集。
    
    Args:
        batch_data: dict, 包含 'images', 'depths', 'Ks', 'T_0to', 'T', 'scales', 'all_5view_ids'
        subset_views_list: list[int], e.g., [5,4,3,2]
        scale: int, 下采样比例
    
    Returns:
        batch_points_dict: dict, k -> (subset_ids, (multi_corrs, vis))
    """
    multi_corrs_5view, vis_5view = generate_multi_corrs_from_data(batch_data, scale=scale)

    # ===== 处理 all_ids =====
    all_ids = batch_data['all_5view_ids']
    if isinstance(all_ids, torch.Tensor):
        all_ids = all_ids.cpu().numpy().flatten().tolist()
    all_ids = [int(x) if not isinstance(x, int) else x for x in all_ids]
    id_to_idx = {vid: i for i, vid in enumerate(all_ids)}

    batch_points_dict = {}
    used_points = set()  # 用于记录已经被使用的匹配点坐标 (anchor view)

    for k in subset_views_list:
        if k == 5:
            # 保留全部 5-view 点
            batch_points_dict[5] = (all_ids, (multi_corrs_5view, vis_5view))
            # 添加 anchor 坐标到 used_points，舍入到1位小数避免精度问题
            anchor_coords = multi_corrs_5view[:, 0].cpu().numpy()  # [N, 2]
            used_points.update(tuple(np.round(coord, 1)) for coord in anchor_coords)
        else:
            # 生成互斥子集
            subset_ids, (multi_corrs_k, vis_k) = select_subset_and_recompute_multi_corrs(
                batch_data,
                subset_views=k,
                scale=scale,
                used_points=used_points
            )
           # 如果返回为空，则给空数组
            if multi_corrs_k is None or multi_corrs_k.shape[0] == 0:
                subset_ids, multi_corrs_k, vis_k = [], np.zeros((0, k, 2)), None
            else:
                # 强制 subset_ids 全部为 Python int
                subset_ids = [int(x) if not isinstance(x, int) else x for x in subset_ids]

            batch_points_dict[k] = (subset_ids, (multi_corrs_k, vis_k))

    return batch_points_dict, id_to_idx

# ================================
# 1️⃣ 生成完整 5-view 对应点
# ================================
@torch.no_grad()
def generate_multi_corrs_from_data(data, scale=8, cycle_thresh=1.5):
    """
    从 data (5-view) 中生成 multi-view 对应点坐标。

    Args:
        data: dict, 包含 keys 'images', 'depths', 'Ks', 'T_0to', 'scales'
              每个 key 是 list of length 5
        scale: 下采样比例
        cycle_thresh: cycle consistency 阈值

    Returns:
        multi_corrs: torch.Tensor, [num_points, 5, 2] 对应原图坐标
    """
    device = data['images'][0].device
    B = 1  # 单个 scene

    images = [img.unsqueeze(0) if img.dim()==3 else img for img in data['images']]
    depths = [d.unsqueeze(0) if d.dim()==2 else d for d in data['depths']]
    Ks = [K.unsqueeze(0) if K.dim()==2 else K for K in data['Ks']]
    T_0to = [T.unsqueeze(0) if T.dim()==2 else T for T in data['T_0to']]
    scales = [torch.tensor([s], device=device) if not isinstance(s, torch.Tensor) else s for s in data['scales']]

    # ===== 1. reference grid =====
    _, _, H, W = images[0].shape
    h, w = H // scale, W // scale
    grid = create_meshgrid(h, w, False, device).reshape(1, h*w, 2)
    grid_i = grid * scale  # 原图坐标

    valid_all = torch.ones((1, h*w), dtype=torch.bool, device=device)
    all_points = [grid_i]

    # ===== 2. cycle consistency =====
    V = len(images)
    for i in range(1, V):
        valid_fw, warped_fw = warp_kpts(grid_i, depths[0], depths[i], T_0to[i], Ks[0], Ks[i])
        T_i_to_0 = torch.inverse(T_0to[i])
        valid_bw, warped_bw = warp_kpts(warped_fw, depths[i], depths[0], T_i_to_0, Ks[i], Ks[0])
        dist = torch.norm(grid_i - warped_bw, dim=-1)
        mask_cycle = dist < cycle_thresh
        valid = valid_fw & valid_bw & mask_cycle
        valid_all &= valid
        all_points.append(warped_fw)

    # ===== 3. final filtering =====
    final_mask = valid_all[0]
    if final_mask.sum() == 0:
        empty_corrs = torch.empty((0, V, 2), device=device)
        empty_vis = torch.empty((0, V), dtype=torch.bool, device=device)
        return empty_corrs, empty_vis

    pts = [p[0, final_mask] for p in all_points]
    multi_corrs_5view = torch.stack(pts, dim=1)  # [num_points, 5, 2]

    # ===== 4. scale 回原图 =====
    for i in range(V):
        multi_corrs_5view[:, i, 0] /= scales[i][0]  # x scale
        multi_corrs_5view[:, i, 1] /= scales[i][1]  # y scale
    
    # ===== vis mask（所有点都有效）=====
    vis = torch.ones(multi_corrs_5view.shape[:2], dtype=torch.bool, device=device)

    return multi_corrs_5view, vis

@torch.no_grad()
def select_subset_and_recompute_multi_corrs(data_5view, subset_views=3, scale=4, used_points=None):
    """
    生成 k-view 对应点，保证与 used_points 互斥。
    """
    import numpy as np
    import torch
    from copy import deepcopy

    if used_points is None:
        used_points = set()

    # 统一 all_ids 类型
    all_ids = data_5view['all_5view_ids']
    if isinstance(all_ids, torch.Tensor):
        all_ids = all_ids.cpu().numpy().flatten().tolist()
    all_ids = [int(x.item()) if isinstance(x, torch.Tensor) else int(x) for x in all_ids]

    # 随机选择 subset_views 个 view（保证 anchor 在内）
    anchor_id = all_ids[0]
    remaining_ids = [v for v in all_ids if v != anchor_id]
    if subset_views == 1:
        subset_ids = [anchor_id]
    elif subset_views == 2:
        # 对于2-view，使用前两个（假设是原始pair）
        subset_ids = all_ids[:2]
    else:
        subset_ids = [anchor_id] + list(np.random.choice(remaining_ids, subset_views-1, replace=False))

    # 构造子集 data
    subset_indices = [all_ids.index(v) for v in subset_ids]
    data_subset = deepcopy(data_5view)
    for key in ['images', 'depths', 'Ks', 'T_0to', 'T', 'scales']:
        data_subset[key] = [data_subset[key][i] for i in subset_indices]
    data_subset['all_5view_ids'] = subset_ids

    # 生成 multi_corrs
    multi_corrs_subset, vis_tensor = generate_multi_corrs_from_data(data_subset, scale=scale)

    # ===== 安全互斥处理 =====
    if len(used_points) > 0 and multi_corrs_subset.shape[0] > 0:
        keep_indices = []
        for i in range(multi_corrs_subset.shape[0]):
            anchor_coord = tuple(np.round(multi_corrs_subset[i, 0].cpu().numpy(), 1))
            if anchor_coord not in used_points:
                keep_indices.append(i)
                used_points.add(anchor_coord)  # 添加到 used_points 以防后续子集使用
        
        if keep_indices:
            multi_corrs_subset = multi_corrs_subset[keep_indices]
            vis_tensor = vis_tensor[keep_indices]
        else:
            multi_corrs_subset = torch.empty((0, subset_views, 2), device=multi_corrs_subset.device)
            vis_tensor = torch.empty((0, subset_views), dtype=torch.bool, device=vis_tensor.device)

    return subset_ids, (multi_corrs_subset, vis_tensor)
"""
    multi_coors: [N, 5, 2]
    N 是匹配点的数量
    
    multi_corrs[i] = [
    [x0, y0],   # image0 (anchor)
    [x1, y1],   # image1
    [x2, y2],   # image2
    [x3, y3],   # image3
    [x4, y4],   # image4
]

"""
"""
batch_5 = dataset[idx]  # 5-view 原始数据

# 从5-view里随机选4-view，并重新计算对应点
batch_4, multi_corrs_4 = select_subset_and_recompute_multi_corrs(batch_5, subset_views=4)

# 从5-view里随机选3-view，并重新计算对应点
batch_3, multi_corrs_3 = select_subset_and_recompute_multi_corrs(batch_5, subset_views=3)

# 从5-view里随机选2-view，并重新计算对应点
batch_2, multi_corrs_2 = select_subset_and_recompute_multi_corrs(batch_5, subset_views=2)

"""
