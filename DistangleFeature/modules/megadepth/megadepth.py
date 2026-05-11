import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
import glob
from modules.megadepth.megadepth_utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net
import numpy.random as rnd

import pdb, tqdm, os

"""
从一系列megadepth 匹配图里得到5个同一个view 下对应点信息的方法如下
选一个 anchor 图 I0：

在 I0 上采样 grid 点
    用 warp_kpts：
        I0 → I1
        I0 → I2
        I0 → I3
        I0 → I4
    对每个点做：
        valid = valid_01 & valid_02 & valid_03 & valid_04
        👉 只保留 在5张图都可见的点

"""



class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score = 0.3, #0.3,
                 max_overlap_score = 0.8, #1,
                 load_depth = True,
                 img_resize = (800,608), #or None
                 df=32,
                 img_padding=False,  
                 depth_padding=True, # MegaDepth 深度图的分辨率不统一
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        
        """
        npz 是某一个scene 的索引文件 里面包含
        {
            'pair_infos': [ (img_i, img_j, overlap_score), ... ],
            'image_paths': [...],
            'depth_paths': [...],
            'intrinsics': [...],
            'poses': [...]
        }
        
        当overlap = 0.1 时表示几乎没有共同区域。overlap = 0.9 时表示视角接近
        需要根据overlap 进行过滤，因为太低overlap 是噪声样本，学不到。而太高的overlap 又太简单，学不到泛化。
        
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]
        self.load_depth = load_depth
        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score]

        # Create graph
        self.graph = defaultdict(list)
        for (i, j), overlap, _ in self.pair_infos:
            self.graph[i].append((j, overlap))
            self.graph[j].append((i, overlap))


        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None #and img_padding and depth_padding

        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.
        self.min_overlap_score = min_overlap_score
        self.max_overlap_score = max_overlap_score
        
        
        # 兼容 D2 Net 的路径格式
        for idx in range(len(self.scene_info['image_paths'])):
            self.scene_info['image_paths'][idx] = fix_path_from_d2net(self.scene_info['image_paths'][idx])

        for idx in range(len(self.scene_info['depth_paths'])):
            self.scene_info['depth_paths'][idx] = fix_path_from_d2net(self.scene_info['depth_paths'][idx])
        
    
    # Sample 5 views from 
    def sample_five_views(self, anchor):
        neighbors = [j for j, o in self.graph[anchor] if o > self.min_overlap_score and o < self.max_overlap_score  ]
        
        
        if len(neighbors) < 4:
            return None

        selected = np.random.choice(neighbors, 4, replace=False)
        return [anchor] + list(selected)


    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx, subset_views=None):
        """
        subset_views: int, optional
            如果为 None，返回 5-view。
            如果为 4/3/2，则从采样到的 5-view 中随机选择 subset_views 个。
        """
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx % len(self)]
    
        anchor = idx0
        # 1. 先采样 5 张 view
        ids_5 = self.sample_five_views(anchor)
        if ids_5 is None:
            return self.__getitem__(np.random.randint(len(self)), subset_views=subset_views)
    
        # 2. 如果需要子集，从5-view里随机选择
        if subset_views is not None and subset_views < len(ids_5):
            ids = list(np.random.choice(ids_5, subset_views, replace=False))
            # 保证 anchor 一定在子集里
            if anchor not in ids:
                ids[0] = anchor
        else:
            ids = ids_5

        # 3. 读取 images / depth / Ks / poses / scales
        images, depths, Ks, poses, scales = [], [], [], [], []

        for i in ids:
            img_path = osp.join(self.root_dir, self.scene_info['image_paths'][i])
            image, mask, scale = read_megadepth_gray(img_path, self.img_resize, self.df, self.img_padding, None)
            images.append(image)
            scales.append(scale)

            if self.load_depth:
                depth = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][i]),
                    pad_to=self.depth_max_size)
                depths.append(depth)

            Ks.append(torch.tensor(self.scene_info['intrinsics'][i], dtype=torch.float).reshape(3, 3))
            poses.append(self.scene_info['poses'][i])

        # 4. 计算相对变换
        T0 = poses[0]
        T_0to = []
        for i in range(len(poses)):
            Ti = poses[i]
            T = torch.tensor(np.matmul(Ti, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]
            T_0to.append(T)

        data = {
            'images': images,
            'depths': depths,
            'Ks': Ks,
            'T_0to': T_0to,
            'T': poses,
            'scales': scales,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'view_ids': ids,
            'all_5view_ids': ids_5  # 可以保留完整5-view的信息
        }
        return data
        