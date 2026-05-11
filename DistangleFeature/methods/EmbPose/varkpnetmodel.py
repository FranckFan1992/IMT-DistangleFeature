import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .utils import *
from methods.Xfeat.xfeat import XFeat


# -------------------------
# Shared Backbone
# -------------------------

class SharedBackbone_XFeat(nn.Module):
    def __init__(self, out_dim=128, freeze=True):
        super().__init__()

        self.xfeat = XFeat()
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Conv2d(64, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True)
        )

        if freeze:
            for p in self.xfeat.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.xfeat.getFeatDesc(x)
        feat = self.proj(feat)
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        return feat




# -------------------------
# Descriptor Encoder
# -------------------------

class TripleDescriptorHead(nn.Module):
    def __init__(self, in_dim=128, dim_inv=128, dim_geo=32, dim_noise=16):
        super().__init__()

        self.inv_head = nn.Sequential(
            nn.Conv2d(in_dim, dim_inv, 3, padding=1),
            nn.GroupNorm(8, dim_inv),
            nn.ReLU(),
            nn.Conv2d(dim_inv, dim_inv, 1)
        )

        self.geo_head = nn.Sequential(
            nn.Conv2d(in_dim, dim_geo, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_geo, dim_geo, 1)
        )

        self.noise_head = nn.Sequential(
            nn.Conv2d(in_dim, dim_noise, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_noise, dim_noise, 1)
        )

    def forward(self, x):
        f_inv = F.normalize(self.inv_head(x), dim=1)
        f_geo = self.geo_head(x)
        f_noise = self.noise_head(x)
        return f_inv, f_geo, f_noise


class GeometryTransform(nn.Module):
    def __init__(self, geo_dim=32, pose_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(geo_dim + pose_dim, 128),
            nn.ReLU(),
            nn.Linear(128, geo_dim)
        )

    def forward(self, f_geo, pose_embed):
        x = torch.cat([f_geo, pose_embed], dim=1)
        delta = self.mlp(x)
        return f_geo + delta


class FeatureDecoder(nn.Module):
    def __init__(self, dim_inv=128, dim_geo=32, dim_noise=16, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_inv + dim_geo + dim_noise, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, f_inv, f_geo, f_noise):
        x = torch.cat([f_inv, f_geo, f_noise], dim=1)
        return self.net(x)


# -------------------------
# Pose Encoder
# -------------------------

class PoseEncoder(nn.Module):
    def __init__(self, pose_dim=16, pose_embed=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pose_embed)
        )

    def forward(self, pose):
        return self.mlp(pose)


class FiLMModulation(nn.Module):
    def __init__(self, pose_dim=128, feat_dim=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim * 2)
        )

    def forward(self, feat, pose_embed):
        gamma_beta = self.mlp(pose_embed)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma * feat + beta


class PointDecoder(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Dense Feature AutoEncoder
# -------------------------

class VUDNet(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 dim_geo=32,
                 dim_noise=16,
                 pose_dim=16,
                 pose_embed=128,
                 backbone: nn.Module = None,
                 backbone_name: str = 'xfeat',
                 freeze_backbone: bool = True,
                 backbone_kwargs: dict = None):
        super().__init__()

        self.feature_dim = feature_dim
        self.backbone_name = backbone_name

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = build_shared_backbone(
                backbone_name,
                out_dim=feature_dim,
                freeze=freeze_backbone,
                **(backbone_kwargs or {}),
            )

        self.encoder = TripleDescriptorHead(
            in_dim=feature_dim,
            dim_inv=feature_dim,
            dim_geo=dim_geo,
            dim_noise=dim_noise
        )

        self.pose_encoder = PoseEncoder(pose_dim, pose_embed)
        self.geo_transform = GeometryTransform(dim_geo, pose_embed)

        self.decoder = FeatureDecoder(
            dim_inv=feature_dim,
            dim_geo=dim_geo,
            dim_noise=dim_noise,
            out_dim=feature_dim
        )

        self.reliability_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        shared = self.backbone(img)
        f_inv, f_geo, f_noise = self.encoder(shared)
        reliability = self.reliability_head(shared)
        heatmap = self.heatmap_head(shared)

        return {
            "shared": shared,
            "f_inv": f_inv,
            "f_geo": f_geo,
            "f_noise": f_noise,
            "f_app": f_noise,
            "reliability": reliability,
            "heatmap": heatmap,
        }

    def transform_geo(self, f_geo, T_i, T_j):
        if T_i.dim() == 2:
            T_i = T_i.unsqueeze(0)
        if T_j.dim() == 2:
            T_j = T_j.unsqueeze(0)

        if T_i.shape[0] == 1 and f_geo.shape[0] > 1:
            T_i = T_i.expand(f_geo.shape[0], -1, -1)
        if T_j.shape[0] == 1 and f_geo.shape[0] > 1:
            T_j = T_j.expand(f_geo.shape[0], -1, -1)

        pose = pose_matrix_to_9d(T_i, T_j)
        pose_embed = self.pose_encoder(pose)
        f_geo_j = self.geo_transform(f_geo, pose_embed)
        return f_geo_j

    def reconstruct_feature(self, f_inv, f_geo, f_noise):
        return self.decoder(f_inv, f_geo, f_noise)

    def predict_view(self, f_inv_i, f_geo_i, f_noise_j, pose_i, pose_j):
        f_geo_j = self.transform_geo(f_geo_i, pose_i, pose_j)
        shared_j = self.reconstruct_feature(
            f_inv_i,
            f_geo_j,
            f_noise_j
        )
        return shared_j, f_geo_j

    def replace_backbone(self,
                         backbone: nn.Module = None,
                         backbone_name: str = None,
                         freeze_backbone: bool = True,
                         backbone_kwargs: dict = None):
        """Replace the shared backbone after training.

        The remaining network parameters stay unchanged.
        The new backbone must output a feature map of shape [B, C, H, W]
        with C equal to self.feature_dim.
        """
        if backbone is None and backbone_name is None:
            raise ValueError('Provide either backbone or backbone_name to replace_backbone.')

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = build_shared_backbone(
                backbone_name,
                out_dim=self.feature_dim,
                freeze=freeze_backbone,
                **(backbone_kwargs or {}),
            )

        self.backbone_name = backbone_name or self.backbone_name


def mutual_information_loss(stable, view, noise):
    return F.mse_loss(stable, view) + F.mse_loss(stable, noise)


def reconstruction_loss(view_features, pose_emb):
    if pose_emb is None:
        return torch.tensor(0.0, device=view_features.device)
    return F.mse_loss(view_features, pose_emb)


def regularization_loss(stable, view, noise, weight=1e-4):
    return weight * (stable.pow(2).mean() + view.pow(2).mean() + noise.pow(2).mean())


