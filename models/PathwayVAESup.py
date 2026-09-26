"""PathwayVAESup

有监督版本的单通路 VAE。
在 PathwayVAE 基础上增加一个年龄预测头，
训练时同时优化：重建损失 + KL散度 + 年龄回归损失。

这样潜在空间 mu 在保留通路甲基化模式的同时，
也被迫保留与年龄相关的信息。
"""

import torch
import torch.nn as nn
import numpy as np


class PathwayVAESup(nn.Module):
    """有监督单通路 VAE（带年龄预测头）。

    参数
    ----
    n_cpgs     : 该通路覆盖的 CpG 数量
    latent_dim : 潜在空间维度（默认 16）
    hidden_layer_encoder_topology : 编码器隐藏层列表，解码器镜像（默认 [64, 64]）
    """

    def __init__(self, n_cpgs: int, latent_dim: int = 16,
                 hidden_layer_encoder_topology=None):
        super().__init__()
        if hidden_layer_encoder_topology is None:
            hidden_layer_encoder_topology = [64, 64]

        self.n_cpgs   = n_cpgs
        self.n_latent = latent_dim

        pre_latent  = [n_cpgs] + hidden_layer_encoder_topology
        post_latent = [latent_dim] + hidden_layer_encoder_topology[::-1]

        # ── 编码器 ──
        enc_layers = []
        for i in range(len(pre_latent) - 1):
            layer = nn.Linear(pre_latent[i], pre_latent[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            enc_layers.append(nn.Sequential(layer, nn.ReLU()))
        self.encoder = nn.Sequential(*enc_layers) if enc_layers else nn.Identity()

        # 潜在层：普通线性映射，不加 BN
        # （BN 会与 KL 散度优化目标冲突，破坏潜在空间的区分性）
        self.z_mean = nn.Linear(pre_latent[-1], latent_dim)
        self.z_var  = nn.Linear(pre_latent[-1], latent_dim)

        # ── 解码器（Beta在[0,1]，输出层用Sigmoid）──
        dec_layers = []
        for i in range(len(post_latent) - 1):
            layer = nn.Linear(post_latent[i], post_latent[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            dec_layers.append(nn.Sequential(layer, nn.ReLU()))
        out_layer = nn.Linear(post_latent[-1], n_cpgs)
        nn.init.xavier_uniform_(out_layer.weight)
        dec_layers.append(nn.Sequential(out_layer, nn.Sigmoid()))
        self.decoder = nn.Sequential(*dec_layers)

        # ── 年龄预测头：mu → 1维年龄（新增）──
        self.age_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1)
        )
        for m in self.age_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def sample_z(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def encode(self, x):
        h      = self.encoder(x)
        mu     = self.z_mean(h)
        logvar = self.z_var(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """返回 (recon, mu, logvar, age_pred)。"""
        mu, logvar = self.encode(x)
        z          = self.sample_z(mu, logvar)
        recon      = self.decode(z)
        age_pred   = self.age_head(mu).squeeze(-1)   # (B,)
        return recon, mu, logvar, age_pred


def vae_sup_loss(
    recon, x, mu, logvar, age_pred, age_true,
    recon_fn, epoch: int,
    kl_warm_up: int = 0,
    beta_kl: float = 1.0,
    alpha_age: float = 1.0,
):
    """有监督 VAE 损失 = 重建损失 + KL散度 + alpha * 年龄回归损失。

    量级对齐：
    - recon_loss: recon_fn(reduction='sum') / batch_size
    - kl_loss: mean(sum(..., dim=1))
    - age_loss: mse_loss(reduction='mean')，量级较小，用 alpha_age 调节

    参数
    ----
    recon_fn   : nn.MSELoss(reduction='sum')
    alpha_age  : 年龄损失权重（默认 1.0，可调）
    """
    batch_size = x.size(0)
    # 对 batch 平均，对特征维度累加
    recon_loss = recon_fn(recon, x) / batch_size

    # KL 散度：特征维度求和，batch 维度求平均
    kl_loss = torch.mean(
        0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=1)
    )
    kl_loss = kl_loss * beta_kl
    if epoch < kl_warm_up:
        kl_loss = kl_loss * float(np.clip(epoch / kl_warm_up, 0.0, 1.0))

    # 年龄回归损失（对归一化后的年龄，mean reduction）
    age_loss = nn.functional.mse_loss(age_pred, age_true.view(-1))

    total = recon_loss + kl_loss + alpha_age * age_loss
    return total, recon_loss, kl_loss, age_loss
