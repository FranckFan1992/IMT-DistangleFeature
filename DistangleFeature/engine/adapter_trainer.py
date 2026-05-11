import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from modules.megadepth.megadepth import MegaDepthDataset
from methods.EmbPose.varkpnetmodel import VUDNet, SharedBackbone_XFeat


"""
python -m engine.adapter_trainer \
  --data_path datasets/MegaDepth_v1 \
  --npz_path datasets/scene_info_0.1_0.7/0022_0.1_0.3.npz \
  --teacher_ckpt checkpoints/kpnet_iter_xfeat_45000.pth \
  --student_backbone r2d2 \
  --student_pretrained \
  --batch_size 8 \
  --epochs 10 \
  --save_dir checkpoints/adapter

"""


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_dim, affine=False)
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class ResNetBackbone(nn.Module):
    def __init__(self, arch='resnet18', out_dim=128, pretrained=True):
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError('torchvision is required for ResNetBackbone.') from exc

        if not hasattr(models, arch):
            raise ValueError(f'Unsupported ResNet arch: {arch}')

        net = getattr(models, arch)(pretrained=pretrained)
        if arch in ['resnet18', 'resnet34']:
            in_channels = 512
        else:
            in_channels = 2048

        self.features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.project = nn.Conv2d(in_channels, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        return self.project(x)


class R2D2Backbone(nn.Module):
    def __init__(self, out_dim=128, pretrained=True, top_k=4096, detection_threshold=0.05, fill_kernel=7):
        super().__init__()
        try:
            from kornia.feature import R2D2
        except ImportError as exc:
            raise ImportError('kornia is required for R2D2Backbone.') from exc

        self.model = R2D2(pretrained=pretrained).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        self.fill_kernel = fill_kernel
        self.out_dim = out_dim

        with torch.no_grad():
            dummy = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
            out = self.model(dummy)
            desc = self._parse_output(out)['descriptors']
            desc_dim = desc.shape[-1]

        self.project = nn.Conv2d(desc_dim, out_dim, kernel_size=1)

    def _parse_output(self, out):
        if isinstance(out, dict):
            keypoints = out.get('keypoints')
            descriptors = out.get('descriptors')
            scores = out.get('scores', None)
        elif isinstance(out, tuple) or isinstance(out, list):
            if len(out) >= 2:
                keypoints, descriptors = out[0], out[1]
                scores = out[2] if len(out) >= 3 else None
            else:
                raise RuntimeError('Unexpected R2D2 output format.')
        else:
            raise RuntimeError('Unexpected R2D2 output format.')

        return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}

    def _sparse_to_dense(self, keypoints, descriptors, H, W):
        B, N, _ = keypoints.shape
        C = descriptors.shape[-1]
        device = descriptors.device

        dense = torch.zeros((B, C, H, W), device=device)
        count = torch.zeros((B, 1, H, W), device=device)

        x = torch.clamp(keypoints[..., 0].round().long(), 0, W - 1)
        y = torch.clamp(keypoints[..., 1].round().long(), 0, H - 1)

        for b in range(B):
            if N == 0:
                continue
            flat_idx = y[b] * W + x[b]
            dense[b] = dense[b].view(C, -1).scatter_add_(1, flat_idx.unsqueeze(0).expand(C, N), descriptors[b].transpose(0, 1)).view(C, H, W)
            count[b] = count[b].view(1, -1).scatter_add_(1, flat_idx.unsqueeze(0), torch.ones((1, N), device=device)).view(1, H, W)

        valid = count > 0
        dense = dense / (count + 1e-6)

        if self.fill_kernel > 1:
            valid = valid.float()
            dense = F.avg_pool2d(dense * valid, kernel_size=self.fill_kernel, stride=1, padding=self.fill_kernel // 2)
            norm = F.avg_pool2d(valid, kernel_size=self.fill_kernel, stride=1, padding=self.fill_kernel // 2)
            dense = dense / (norm + 1e-6)

        return dense

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out = self.model(x)
        parsed = self._parse_output(out)
        keypoints = parsed['keypoints']
        descriptors = parsed['descriptors']

        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(0)
            descriptors = descriptors.unsqueeze(0)

        B, _, H, W = x.shape
        dense = self._sparse_to_dense(keypoints, descriptors, H, W)
        return self.project(dense)


class ImageOnlyDataset(Dataset):
    def __init__(self, base_dataset, view_index=0, transform=None):
        self.base_dataset = base_dataset
        self.view_index = view_index
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = sample['images'][self.view_index]

        if isinstance(image, np.ndarray):
            img = image
        else:
            img = np.array(image)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=0)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.transpose(np.tile(img, (1, 1, 3)), (2, 0, 1))
        elif img.ndim == 3 and img.shape[0] == 3:
            pass
        else:
            raise RuntimeError(f'Unsupported image shape: {img.shape}')

        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        img = torch.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)

        return {'image': img}


def default_collate(batch):
    if isinstance(batch[0], dict):
        return {
            'image': torch.stack([item['image'] for item in batch], dim=0)
        }
    return torch.stack(batch, dim=0)


class VUDNetAdapterWrapper(nn.Module):
    def __init__(
        self,
        vudnet,
        backbone_other,
        adapter_in_dim=None,
        freeze_teacher=True,
        freeze_student_backbone=True,
    ):
        super().__init__()

        self.vudnet = vudnet
        if freeze_teacher:
            self.vudnet.eval()
            for p in self.vudnet.parameters():
                p.requires_grad = False

        self.backbone_other = backbone_other
        if freeze_student_backbone:
            self.backbone_other.eval()
            for p in self.backbone_other.parameters():
                p.requires_grad = False

        if adapter_in_dim is None:
            adapter_in_dim = self._infer_backbone_feature_dim()

        self.adapter = FeatureAdapter(
            in_dim=adapter_in_dim,
            out_dim=self.vudnet.feature_dim,
        )

    def _infer_backbone_feature_dim(self):
        self.backbone_other.eval()
        with torch.no_grad():
            device = next(self.backbone_other.parameters()).device if any(self.backbone_other.parameters()) else torch.device('cpu')
            dummy = torch.zeros((1, 3, 256, 256), device=device)
            feat = self.backbone_other(dummy)
            if not isinstance(feat, torch.Tensor):
                raise RuntimeError('backbone_other.forward must return a tensor feature map.')
            return feat.shape[1]

    def forward(self, img):
        # teacher: 已训练好的 VUDNet（以 XFeat 特征为 backbone）
        with torch.no_grad():
            teacher_out = self.vudnet(img)
            F_x = teacher_out["shared"]
            f_inv_x = teacher_out["f_inv"]
            f_geo_x = teacher_out["f_geo"]

        # student: 其他 backbone + adapter
        F_y = self.backbone_other(img)
        if F_y.shape[-2:] != F_x.shape[-2:]:
            F_y = F.interpolate(F_y, size=F_x.shape[-2:], mode='bilinear', align_corners=False)
        F_hat = self.adapter(F_y)

        f_inv_hat, f_geo_hat, f_noise_hat = self.vudnet.encoder(F_hat)
        recon_hat = self.vudnet.reconstruct_feature(f_inv_hat, f_geo_hat, f_noise_hat)

        return {
            "F_x": F_x,
            "F_hat": F_hat,
            "f_inv_x": f_inv_x,
            "f_geo_x": f_geo_x,
            "f_inv_hat": f_inv_hat,
            "f_geo_hat": f_geo_hat,
            "f_noise_hat": f_noise_hat,
            "recon_hat": recon_hat,
        }


def distillation_loss(outputs,
                       w_feat=1.0,
                       w_cos=0.5,
                       w_inv=0.5,
                       w_geo=0.5,
                       w_recon=0.2,
                       w_stat=0.1):
    F_x = outputs["F_x"]
    F_hat = outputs["F_hat"]
    F_x_norm = F.normalize(F_x, dim=1)
    F_hat_norm = F.normalize(F_hat, dim=1)

    f_inv_x = outputs["f_inv_x"]
    f_geo_x = outputs["f_geo_x"]
    f_inv_hat = outputs["f_inv_hat"]
    f_geo_hat = outputs["f_geo_hat"]
    recon_hat = outputs["recon_hat"]

    loss_feat = F.l1_loss(F_hat_norm, F_x_norm)
    loss_cos = 1 - F.cosine_similarity(
        F_hat_norm.flatten(1),
        F_x_norm.flatten(1),
        dim=1
    ).mean()
    loss_inv = F.l1_loss(f_inv_hat, f_inv_x)
    loss_geo = F.l1_loss(f_geo_hat, f_geo_x)
    loss_recon = F.l1_loss(recon_hat, F_x)

    def stats(x):
        return x.mean(dim=[2, 3]), x.std(dim=[2, 3])

    mean_x, std_x = stats(F_x)
    mean_hat, std_hat = stats(F_hat)
    loss_stat = F.l1_loss(mean_hat, mean_x) + F.l1_loss(std_hat, std_x)

    loss = (
        w_feat * loss_feat +
        w_cos * loss_cos +
        w_inv * loss_inv +
        w_geo * loss_geo +
        w_recon * loss_recon +
        w_stat * loss_stat
    )

    return loss, {
        "loss_feat": loss_feat.item(),
        "loss_cos": loss_cos.item(),
        "loss_inv": loss_inv.item(),
        "loss_geo": loss_geo.item(),
        "loss_recon": loss_recon.item(),
        "loss_stat": loss_stat.item(),
    }


class AdapterTrainer:
    def __init__(self,
                 model,
                 dataloader,
                 lr=1e-3,
                 weight_decay=0.0,
                 scheduler=None,
                 device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.adapter.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = scheduler

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        count = 0
        metrics = {
            'loss_feat': 0.0,
            'loss_cos': 0.0,
            'loss_inv': 0.0,
            'loss_geo': 0.0,
            'loss_recon': 0.0,
            'loss_stat': 0.0,
        }

        for batch in self.dataloader:
            img = batch if isinstance(batch, torch.Tensor) else batch['image']
            img = img.to(self.device)

            outputs = self.model(img)
            loss, log = distillation_loss(outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            for k, v in log.items():
                metrics[k] += v
            count += 1

        if count == 0:
            return 0.0, {k: 0.0 for k in metrics}

        return total_loss / count, {k: v / count for k, v in metrics.items()}

    def train(self, epochs, save_dir=None, save_interval=1):
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

        for epoch in range(1, epochs + 1):
            avg_loss, avg_metrics = self.train_epoch()
            log_line = (
                f'Epoch {epoch}/{epochs} | loss={avg_loss:.6f} ' +
                ' '.join([f'{k}={v:.6f}' for k, v in avg_metrics.items()])
            )
            print(log_line)

            if save_dir is not None and epoch % save_interval == 0:
                save_path = os.path.join(save_dir, f'adapter_epoch_{epoch}.pth')
                torch.save(self.model.adapter.state_dict(), save_path)
                print(f'Saved adapter checkpoint: {save_path}')


def build_adapter_trainer(vudnet, backbone_other, dataloader, adapter_in_dim=None, device=None):
    wrapper = VUDNetAdapterWrapper(
        vudnet=vudnet,
        backbone_other=backbone_other,
        adapter_in_dim=adapter_in_dim,
        freeze_teacher=True,
        freeze_student_backbone=True,
    )
    return AdapterTrainer(
        model=wrapper,
        dataloader=dataloader,
        device=device,
    )


def inference(model, img):
    with torch.no_grad():
        F_y = model.backbone_other(img)
        F_hat = model.adapter(F_y)
        out = model.vudnet.encoder(F_hat)
    return out


def create_teacher_vudnet(checkpoint_path=None,
                          feature_dim=128,
                          dim_geo=32,
                          dim_noise=16,
                          pose_dim=16,
                          pose_embed=128,
                          device=None):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_backbone = SharedBackbone_XFeat(out_dim=feature_dim, freeze=True).to(device)
    teacher = VUDNet(
        feature_dim=feature_dim,
        dim_geo=dim_geo,
        dim_noise=dim_noise,
        pose_dim=pose_dim,
        pose_embed=pose_embed,
        backbone=teacher_backbone,
        freeze_backbone=False,
    ).to(device)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        teacher.load_state_dict(state, strict=False)
        print(f'Loaded teacher VUDNet from: {checkpoint_path}')

    teacher.eval()
    return teacher


def build_student_backbone(backbone_type, feature_dim=128, pretrained=True, r2d2_top_k=4096, r2d2_detection_threshold=0.05):
    if backbone_type.startswith('resnet'):
        return ResNetBackbone(arch=backbone_type, out_dim=feature_dim, pretrained=pretrained)
    if backbone_type == 'r2d2':
        return R2D2Backbone(
            out_dim=feature_dim,
            pretrained=pretrained,
            top_k=r2d2_top_k,
            detection_threshold=r2d2_detection_threshold,
        )

    raise ValueError(f'Unsupported student backbone: {backbone_type}')


def build_dataloader(data_path, npz_path, batch_size, num_workers):
    base_dataset = MegaDepthDataset(
        root_dir=data_path,
        npz_path=npz_path,
        mode='train',
        img_resize=(800, 608),
        df=32,
        img_padding=False,
        depth_padding=False,
    )
    image_dataset = ImageOnlyDataset(base_dataset)
    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=default_collate,
        drop_last=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train adapter for VUDNet feature decomposition')
    parser.add_argument('--data_path', type=str, required=True, help='MegaDepth root directory')
    parser.add_argument('--npz_path', type=str, required=True, help='Scene npz file for MegaDepth dataset')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Pretrained VUDNet teacher checkpoint path')
    parser.add_argument('--student_backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'r2d2'], help='Student backbone type, e.g. resnet18 or r2d2')
    parser.add_argument('--student_pretrained', action='store_true', help='Use pretrained weights for student backbone')
    parser.add_argument('--r2d2_top_k', type=int, default=4096, help='Top-k keypoints for R2D2 sparse feature extraction')
    parser.add_argument('--r2d2_detection_threshold', type=float, default=0.05, help='Detection threshold for R2D2 sparse keypoints')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension for VUDNet and adapter output')
    parser.add_argument('--dim_geo', type=int, default=32, help='Geometry descriptor dimension')
    parser.add_argument('--dim_noise', type=int, default=16, help='Noise descriptor dimension')
    parser.add_argument('--pose_dim', type=int, default=16, help='Pose input dimension')
    parser.add_argument('--pose_embed', type=int, default=128, help='Pose embedding dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for adapter training')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for adapter optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker count')
    parser.add_argument('--save_dir', type=str, default='checkpoints/adapter', help='Directory to save adapter checkpoints')
    parser.add_argument('--save_interval', type=int, default=1, help='Epoch interval for saving adapter checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher = create_teacher_vudnet(
        checkpoint_path=args.teacher_ckpt,
        feature_dim=args.feature_dim,
        dim_geo=args.dim_geo,
        dim_noise=args.dim_noise,
        pose_dim=args.pose_dim,
        pose_embed=args.pose_embed,
        device=device,
    )

    student_backbone = build_student_backbone(
        backbone_type=args.student_backbone,
        feature_dim=args.feature_dim,
        pretrained=args.student_pretrained,
        r2d2_top_k=args.r2d2_top_k,
        r2d2_detection_threshold=args.r2d2_detection_threshold,
    ).to(device)

    dataloader = build_dataloader(
        data_path=args.data_path,
        npz_path=args.npz_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = build_adapter_trainer(
        vudnet=teacher,
        backbone_other=student_backbone,
        dataloader=dataloader,
        device=device,
    )

    print(f'Starting adapter training: student_backbone={args.student_backbone}, batch_size={args.batch_size}, epochs={args.epochs}')
    trainer.train(args.epochs, save_dir=args.save_dir, save_interval=args.save_interval)


if __name__ == '__main__':
    main()
