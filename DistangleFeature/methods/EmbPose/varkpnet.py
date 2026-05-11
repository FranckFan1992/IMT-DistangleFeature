import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from .varkpnetmodel import VarianceKPNetModel
from .interpolator import InterpolateSparse2d

class VarianceKPNet(nn.Module):
    
    def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../../checkpoints/xfeat.pt', top_k = 4096, detection_threshold=0.05):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = VarianceKPNetModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        
        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_state_dict(torch.load(weights, map_location=self.dev))
            else:
                self.net.load_state_dict(weights)
        
        self.interpolator = InterpolateSparse2d('bicubic')
        
    def detectAndCompute(self, x, top_k = None, detection_threshold = None):
        """
			Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return:
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
        """
        if top_k is None: top_k = self.top_k
        if detection_threshold is None: detection_threshold = self.detection_threshold
        x, rh1, rw1 = self.preprocess_tensor(x)
        
        B, _, _H1, _W1 = x.shape
        _, variance_map, desc_map, reliability_map = self.net(x)
        desc_map = F.normalize(desc_map, dim=1)

		#Convert logits to heatmap and extract kpts
        mkpts = self.NMS(reliability_map, threshold=detection_threshold, kernel_size=5)
        scale= 4
        mkpts = mkpts * 4

		#Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        
        var_sampled = _bilinear(variance_map, mkpts, _H1, _W1).squeeze(-1)

        # 归一化到 [0,1]
        var_norm = (var_sampled - var_sampled.min()) / (var_sampled.max() - var_sampled.min() + 1e-6)

        # 转成越小越好
        score_var = 1.0 - var_norm
        
        scores = (_nearest(reliability_map, mkpts, _H1, _W1).squeeze(-1))*score_var
        #scores = _nearest(reliability_map, mkpts, _H1, _W1).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

		#Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :top_k]
        mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

		#Interpolate descriptors at kpts positions
        feats = self.interpolator(desc_map, mkpts, H = _H1, W = _W1)

		#L2-Normalize
        feats = F.normalize(feats, dim=-1)

		#Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)
        
        valid = scores > 0
        return [  
				   {'keypoints': mkpts[b][valid[b]],
					'scores': scores[b][valid[b]],
					'descriptors': feats[b][valid[b]]} for b in range(B) 
			   ]
    
    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = torch.tensor(x).permute(2,0,1)[None]
            elif len(x.shape) == 2:
                x = torch.tensor(x[..., None]).permute(2,0,1)[None]
            else:
                raise RuntimeError('For numpy arrays, only (H,W) or (H,W,C) format is supported.')
        
        if len(x.shape) != 4:
            raise RuntimeError('Input tensor needs to be in (B,C,H,W) format')
        
        x = x.to(self.dev).float()
        
        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W
        
        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw
    
    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]
        
        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)
        
        #Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]
        
        return pos
    
    def match(self, feats1, feats2, min_cossim = 0.82):
        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()
        
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)
        
        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0
        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]
        
        return idx0, idx1

        
        