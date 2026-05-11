import argparse
import os
import torch
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import glob
import tqdm
import torch.nn as nn
import torch.nn.functional as F

from modules.megadepth.megadepth import MegaDepthDataset
from modules.megadepth.megadepth_wraper import *
from modules.sfm_loader import *
from modules.utils import *

from methods.EmbPose.varkpnetmodel import VUDNet

from methods.EmbPose.loss import *

import numpy as np

""""
python -m engine.trainer --data_path datasets/MegaDepth_v1 --cpkt_save_path checkpoints/ --num_iters 50000 --batch_size 2

"""
def to_numpy_image(img):
    """
    统一把输入转成 (H, W, 3)
    支持:
    [B,C,H,W], [C,H,W], [H,W]
    """

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # ===== 1. 去 batch =====
    if img.ndim == 4:  # [B,C,H,W]
        img = img[0]

    # ===== 2. CHW -> HWC =====
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]  # -> (H,W)
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # -> (H,W,3)

    # ===== 3. 灰度 -> RGB =====
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    return img


def plot_multi_view_matches(images, multi_corrs, vis=None, max_points=50, save_path=None):
    """
    可视化多视图匹配点轨迹（严格可见性过滤）
    
    Args:
        images: list of images [C,H,W] 或 [H,W] 或 torch.Tensor
        multi_corrs: [N, V, 2] tensor 或 numpy array
        vis: [N, V] bool，可见性掩码，如果 None，则默认全部可见
        max_points: 最大显示点数
        save_path: 保存路径，如果 None 则 plt.show()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # ===== 转 numpy image =====
    def to_numpy_image(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 4:  # [B,C,H,W]
            img = img[0]
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
            elif img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        return img

    imgs = [to_numpy_image(img) for img in images]
    V = len(imgs)

    H_list = [img.shape[0] for img in imgs]
    W_list = [img.shape[1] for img in imgs]

    H = max(H_list)
    W = sum(W_list)

    canvas = np.zeros((H, W, 3), dtype=imgs[0].dtype)
    offsets = []
    cur_w = 0
    for img in imgs:
        h, w = img.shape[:2]
        canvas[:h, cur_w:cur_w+w] = img
        offsets.append(cur_w)
        cur_w += w

    # ===== 转 numpy 数据 =====
    if isinstance(multi_corrs, torch.Tensor):
        multi_corrs = multi_corrs.detach().cpu().numpy()
    if vis is None:
        vis = np.ones((multi_corrs.shape[0], V), dtype=bool)
    elif isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()

    N = multi_corrs.shape[0]
    if N == 0:
        print("No correspondences")
        return

    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        multi_corrs = multi_corrs[idx]
        vis = vis[idx]
        N = max_points

    # ===== 绘制 =====
    plt.figure(figsize=(15, 5))
    plt.imshow(canvas)
    colors = plt.cm.jet(np.linspace(0, 1, N))

    for i in range(N):
        # 检查该点在所有视图中是否都可见且在图像内
        keep_track = True
        for j in range(V):
            x, y = multi_corrs[i, j]
            h, w = imgs[j].shape[:2]
            if not vis[i, j] or x < 0 or x >= w or y < 0 or y >= h:
                keep_track = False
                break

        if not keep_track:
            continue  # 只保留完全可见的轨迹

        # 画点和连线
        prev_pt = None
        for j in range(V):
            x, y = multi_corrs[i, j]
            plt.scatter(x + offsets[j], y, c=[colors[i]], s=20)
            if prev_pt is not None:
                x0, y0, j0 = prev_pt
                plt.plot([x0 + offsets[j0], x + offsets[j]], [y0, y], c=colors[i], linewidth=1)
            prev_pt = (x, y, j)

    plt.axis('off')
    plt.title(f"Multi-view Correspondences ({V} views)")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=True)
        
def visualize_multi_view_matches(batch_data, batch_points_dict, id_to_idx):
    """
    可视化多视图匹配子集
    """
    images_vis = []
    for img in batch_data['images']:
        if isinstance(img, torch.Tensor) and img.dim() == 4:
            images_vis.append(img[0])
        else:
            images_vis.append(img)

    for k in sorted(batch_points_dict.keys(), reverse=True):
        subset_ids, (corrs_k, vis_k) = batch_points_dict[k]
        if corrs_k.shape[0] == 0:
            print(f"[VIS] No {k}-view points")
            continue

        # 图像顺序和 subset 对齐
        imgs_k = [images_vis[id_to_idx[i]] for i in subset_ids]

        if vis_k is not None:
            vis_k = vis_k.astype(bool) if isinstance(vis_k, np.ndarray) else vis_k.bool()

        plot_multi_view_matches(
            images=imgs_k,
            multi_corrs=corrs_k,
            vis=vis_k,
            max_points=50,
            save_path=None
        )
    
def sfm_collate_fn(batch):
    """
    batch: list of (data0, data1, matches)
    """
    batch_data0 = {}
    batch_data1 = {}
    matches_list = []

    # 所有 key
    for key in batch[0][0]:  # data0 keys
        batch_data0[key] = torch.stack([torch.tensor(b[0][key]) for b in batch])

    for key in batch[0][1]:  # data1 keys
        batch_data1[key] = torch.stack([torch.tensor(b[1][key]) for b in batch])

    for b in batch:
        matches_list.append(torch.tensor(b[2]))  # variable-length, keep list

    return batch_data0, batch_data1, matches_list



class TrainerMultiView:
    def __init__(self, kpnet, datapath, cpkt_save_path, num_iters, batch_size=1, top_k=4096, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kpnet = kpnet.to(self.device)
        self.batch_size = batch_size
        

        #self.dataset = SfMDataset(points, images, camera_infos, datapath)
        megadepth_datapath = datapath
        npz_root = os.path.join(datapath, "..", "scene_info_0.1_0.7")
        #npzpaths = glob.glob(os.path.join(npz_root, '*.npz'))[:]
        
        npzpaths = [os.path.join(npz_root, '0022_0.1_0.3.npz'),
                    os.path.join(npz_root, '0022_0.3_0.5.npz'),
                    os.path.join(npz_root, '0022_0.5_0.7.npz')]
        self.dataset = torch.utils.data.ConcatDataset([
            MegaDepthDataset(root_dir=megadepth_datapath, npz_path=path)
            for path in tqdm.tqdm(npzpaths, desc="[MegaDepth] Loading metadata")
        ])
        
        
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.kpnet.parameters()), lr=3e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30000, gamma=0.5)
        
        
        
        self.num_iters = num_iters
        self.cpkt_save_path = cpkt_save_path
        self.top_k = top_k
        
        self.progress_bar = tqdm.tqdm(range(0, self.num_iters), desc="Training progress")
        self.writer = SummaryWriter(cpkt_save_path + f'/logdir/scr_kpdect_' + time.strftime("%Y_%m_%d-%H_%M_%S"))

    def _get_sample_data(self, batch_data, b):
        sample_data = {}
        for key, value in batch_data.items():
            if isinstance(value, list):
                if len(value) == 0:
                    sample_data[key] = []
                elif isinstance(value[0], torch.Tensor):
                    sample_data[key] = [v[b] for v in value]
                else:
                    sample_data[key] = value[b]
            elif isinstance(value, torch.Tensor):
                sample_data[key] = value[b]
            else:
                try:
                    sample_data[key] = value[b]
                except Exception:
                    sample_data[key] = value
        return sample_data

    def train_iters(self):
        self.kpnet.train()
        data_iter = iter(self.data_loader)
        subset_views_list = [5, 4, 3, 2]

        for iter_idx in range(self.num_iters):
            try:
                batch_data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch_data = next(data_iter)

            images_batch = batch_data['images']
            if isinstance(images_batch, torch.Tensor) and images_batch.dim() == 5:
                batch_size = images_batch.shape[0]
            elif isinstance(images_batch, list) and len(images_batch) > 0:
                if isinstance(images_batch[0], torch.Tensor) and images_batch[0].dim() == 4:
                    batch_size = images_batch[0].shape[0]
                else:
                    batch_size = len(images_batch[0]) if len(images_batch[0]) > 0 else 1
            else:
                batch_size = 1

            self.optimizer.zero_grad()

            subset_weights = {5: 1.0, 4: 1.0, 3: 1.0, 2: 1.0}
            loss_total_accum = 0.0
            valid_batch = 0
            metric_sums = {
                "total": 0.0,
                "inv_nce": 0.0,
                "inv_consistency": 0.0,
                "reconstruction": 0.0,
                "geo": 0.0,
                "cross_reconstruction": 0.0,
                "orthogonality": 0.0,
                "noise_regularization": 0.0,
                "reliability": 0.0,
                "heatmap": 0.0,
            }

            for b in range(batch_size):
                sample_data = self._get_sample_data(batch_data, b)
                sample_images = sample_data['images']
                H_orig, W_orig = sample_images[0].shape[1:]

                batch_points_dict, id_to_idx = generate_exclusive_subsets(sample_data)

                V = len(sample_images)
                shared_maps, f_inv_maps, f_geo_maps, f_noise_maps = [], [], [], []
                rel_maps, heatmap_maps = [], []
                for v in range(V):
                    img = sample_images[v]
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    out = self.kpnet(img.to(self.device))
                    shared_maps.append(out["shared"])
                    f_inv_maps.append(out["f_inv"])
                    f_geo_maps.append(out["f_geo"])
                    f_noise_maps.append(out["f_noise"])
                    rel_maps.append(out["reliability"])
                    heatmap_maps.append(out["heatmap"])

                for k in subset_views_list:
                    subset_ids, (corrs_k, vis_k) = batch_points_dict[k]
                    if len(subset_ids) == 0:
                        continue

                    if not isinstance(corrs_k, torch.Tensor):
                        corrs_k = torch.as_tensor(corrs_k, dtype=torch.float32)
                    N_points = corrs_k.shape[0]
                    if N_points < 20:
                        continue

                    max_points = 5000
                    if N_points > max_points:
                        idx = torch.randperm(N_points)[:max_points]
                        corrs_k = corrs_k[idx]
                        if vis_k is not None:
                            vis_k = vis_k[idx]

                    shared_per_point = []
                    f_inv_per_point = []
                    f_geo_per_point = []
                    f_noise_per_point = []
                    rel_per_point = []
                    subset_poses = []

                    for v in range(k):
                        view_idx = id_to_idx[subset_ids[v]]
                        coords = corrs_k[:, v, :].to(self.device)

                        shared_per_point.append(
                            sample_map_at_coords(shared_maps[view_idx], coords, H_orig, W_orig)
                        )
                        f_inv_per_point.append(
                            sample_map_at_coords(f_inv_maps[view_idx], coords, H_orig, W_orig)
                        )
                        f_geo_per_point.append(
                            sample_map_at_coords(f_geo_maps[view_idx], coords, H_orig, W_orig)
                        )
                        f_noise_per_point.append(
                            sample_map_at_coords(f_noise_maps[view_idx], coords, H_orig, W_orig)
                        )
                        rel_per_point.append(
                            sample_map_at_coords(rel_maps[view_idx], coords, H_orig, W_orig)
                        )
                        subset_poses.append(
                            torch.as_tensor(sample_data['T'][view_idx], dtype=torch.float32, device=self.device)
                        )

                    shared = torch.stack(shared_per_point, dim=1)
                    f_inv = torch.stack(f_inv_per_point, dim=1)
                    f_geo = torch.stack(f_geo_per_point, dim=1)
                    f_noise = torch.stack(f_noise_per_point, dim=1)
                    rel = torch.stack(rel_per_point, dim=1)
                    visibility = (
                        torch.ones((N_points, k), device=self.device, dtype=torch.bool)
                        if vis_k is None else vis_k.bool().to(self.device)
                    )

                    losses = total_disentangle_loss(
                        self.kpnet,
                        f_inv=f_inv,
                        f_geo=f_geo,
                        f_noise=f_noise,
                        shared=shared,
                        poses=subset_poses,
                        visibility=visibility,
                    )
                    loss_rel, _ = reliability_loss_from_confidence(rel, f_inv, visibility)
                    heatmap_gt = build_heatmap_target(
                        corrs_k,
                        visibility,
                        H=H_orig,
                        W=W_orig,
                        downsample=4,
                        device=self.device
                    )
                    loss_heatmap = f_inv.new_tensor(0.0)
                    for v in range(k):
                        view_idx = id_to_idx[subset_ids[v]]
                        loss_heatmap = loss_heatmap + heatmap_loss(
                            heatmap_maps[view_idx],
                            heatmap_gt[v:v+1]
                        )
                    loss_heatmap = loss_heatmap / k

                    losses["reliability"] = loss_rel
                    losses["heatmap"] = loss_heatmap
                    losses["total"] = losses["total"] + 0.1 * loss_rel + 0.5 * loss_heatmap

                    loss_total_accum = loss_total_accum + subset_weights[k] * losses["total"]
                    for key in metric_sums:
                        metric_sums[key] = metric_sums[key] + losses[key].detach()
                    valid_batch += 1

            if valid_batch > 0:
                loss = loss_total_accum / valid_batch
                metric_avgs = {k: v / valid_batch for k, v in metric_sums.items()}
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                metric_avgs = {k: torch.tensor(0.0, device=self.device) for k in metric_sums}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kpnet.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.progress_bar.set_description(
                f"[Iter {iter_idx}] "
                f"loss:{loss.item():.4f} "
                f"inv:{metric_avgs['inv_nce'].item():.4f} "
                f"rel:{metric_avgs['reliability'].item():.4f} "
                f"hm:{metric_avgs['heatmap'].item():.4f} "
                f"geo:{metric_avgs['geo'].item():.4f} "
                f"recon:{metric_avgs['reconstruction'].item():.4f}"
            )

            if (iter_idx + 1) % 5000 == 0 and self.cpkt_save_path is not None:
                torch.save(self.kpnet.state_dict(), f"{self.cpkt_save_path}/kpnet_iter_{iter_idx}.pth")

            self.progress_bar.update(1)

            self.writer.add_scalar('Loss/total', loss.item(), iter_idx)
            self.writer.add_scalar('Loss/inv_nce', metric_avgs['inv_nce'].item(), iter_idx)
            self.writer.add_scalar('Loss/inv_consistency', metric_avgs['inv_consistency'].item(), iter_idx)
            self.writer.add_scalar('Loss/reconstruction', metric_avgs['reconstruction'].item(), iter_idx)
            self.writer.add_scalar('Loss/geo', metric_avgs['geo'].item(), iter_idx)
            self.writer.add_scalar('Loss/cross_reconstruction', metric_avgs['cross_reconstruction'].item(), iter_idx)
            self.writer.add_scalar('Loss/orthogonality', metric_avgs['orthogonality'].item(), iter_idx)
            self.writer.add_scalar('Loss/noise_regularization', metric_avgs['noise_regularization'].item(), iter_idx)
            self.writer.add_scalar('Loss/reliability', metric_avgs['reliability'].item(), iter_idx)
            self.writer.add_scalar('Loss/heatmap', metric_avgs['heatmap'].item(), iter_idx)



                
            
            
            


            
            
            
                
            

                
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-view keypoint network")
    parser.add_argument("--data_path", type=str, default="datasets/MegaDepth_v1/", help="path to training data")
    parser.add_argument("--cpkt_save_path", type=str, default="checkpoints/", help="checkpoint save directory")
    parser.add_argument("--num_iters", type=int, default=50000, help="number of training iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
    args = parser.parse_args()

    variance_kpnet = VUDNet(feature_dim=128,
                 dim_geo=32,
                 dim_noise=16,
                 pose_dim=9,
                 pose_embed=128)
    trainer = TrainerMultiView(
        variance_kpnet,
        args.data_path,
        args.cpkt_save_path,
        args.num_iters,
        batch_size=args.batch_size
    )
    trainer.train_iters()


