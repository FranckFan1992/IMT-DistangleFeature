import torch
import torch.nn as nn


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
    

class VUDNetAdapterWrapper(nn.Module):
    def __init__(self, vudnet, backbone_other):
        super().__init__()

        self.vudnet = vudnet.eval()   # 冻结
        for p in self.vudnet.parameters():
            p.requires_grad = False

        self.backbone_other = backbone_other
        self.adapter = FeatureAdapter(
            in_dim=vudnet.feature_dim,
            out_dim=vudnet.feature_dim
        )

    def forward(self, img, img_aug=None):
        # ----- teacher (xfeat) -----
        with torch.no_grad():
            teacher_out = self.vudnet(img)
            F_x = teacher_out["shared"]
            f_inv_x = teacher_out["f_inv"]
            f_geo_x = teacher_out["f_geo"]

        # ----- student -----
        F_y = self.backbone_other(img)
        F_hat = self.adapter(F_y)

        student_out = self.vudnet.encoder(F_hat)
        f_inv_hat, f_geo_hat, f_noise_hat = student_out

        return {
            "F_x": F_x,
            "F_hat": F_hat,
            "f_inv_x": f_inv_x,
            "f_geo_x": f_geo_x,
            "f_inv_hat": f_inv_hat,
            "f_geo_hat": f_geo_hat,
        }

import torch.nn.functional as F

def distillation_loss(outputs):
    F_x = outputs["F_x"]
    F_hat = outputs["F_hat"]
    
    F_x = F.normalize(F_x, dim=1)
    F_hat = F.normalize(F_hat, dim=1)

    f_inv_x = outputs["f_inv_x"]
    f_geo_x = outputs["f_geo_x"]

    f_inv_hat = outputs["f_inv_hat"]
    f_geo_hat = outputs["f_geo_hat"]

    # ---------- 1. feature 对齐 ----------
    loss_feat = F.l1_loss(F_hat, F_x)

    # ---------- 2. cosine 对齐 ----------
    loss_cos = 1 - F.cosine_similarity(
        F_hat.flatten(1),
        F_x.flatten(1),
        dim=1
    ).mean()

    # ---------- 3. disentangle 对齐 ----------
    loss_inv = F.l1_loss(f_inv_hat, f_inv_x)
    loss_geo = F.l1_loss(f_geo_hat, f_geo_x)

    # ---------- 4. 统计对齐 ----------
    def stats(x):
        return x.mean(dim=[2,3]), x.std(dim=[2,3])

    mean_x, std_x = stats(F_x)
    mean_hat, std_hat = stats(F_hat)

    loss_stat = F.l1_loss(mean_hat, mean_x) + F.l1_loss(std_hat, std_x)

    # ---------- 总 loss ----------
    loss = (
        1.0 * loss_feat +
        0.5 * loss_cos +
        0.5 * loss_inv +
        0.5 * loss_geo +
        0.1 * loss_stat
    )

    return loss, {
        "loss_feat": loss_feat.item(),
        "loss_cos": loss_cos.item(),
        "loss_inv": loss_inv.item(),
        "loss_geo": loss_geo.item(),
        "loss_stat": loss_stat.item(),
    }

def train_adapter():
    model = VUDNetAdapterWrapper(vudnet, backbone_other).cuda()

    optimizer = torch.optim.Adam(
    model.adapter.parameters(),
    lr=1e-3
    )

    for img in dataloader:
        img = img.cuda()

        outputs = model(img)

        loss, log = distillation_loss(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def inference(model, img):
    with torch.no_grad():
        F_y = model.backbone_other(img)
        F_hat = model.adapter(F_y)

        out = model.vudnet.encoder(F_hat)

    return out