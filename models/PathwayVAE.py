"""PathwayVAE

为单个 Reactome 通路设计的 VAE。

输入 : 该通路覆盖的 CpG 位点 Beta 值（原始 [0,1]，未标准化）
输出 :
    recon   : 重建 Beta 值 (B, n_cpgs)  -- Sigmoid 约束
    mu      : 潜在均值     (B, latent_dim)
    logvar  : 潜在对数方差 (B, latent_dim)
    z       : 重参数化采样 (B, latent_dim)
"""

import torch
import torch.nn as nn
import numpy as np


class PathwayVAE(nn.Module):
    """单通路 VAE。

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

        # 编码器（Xavier init + ReLU）
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

        # 解码器（Xavier init + ReLU + Sigmoid 输出）
        dec_layers = []
        for i in range(len(post_latent) - 1):
            layer = nn.Linear(post_latent[i], post_latent[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            dec_layers.append(nn.Sequential(layer, nn.ReLU()))
        out_layer = nn.Linear(post_latent[-1], n_cpgs)
        nn.init.xavier_uniform_(out_layer.weight)
        dec_layers.append(nn.Sequential(out_layer, nn.Sigmoid()))
        self.decoder = nn.Sequential(*dec_layers)

    def sample_z(self, mean, logvar):
        """重参数化：训练时加噪声，推理时直接用均值。"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def encode(self, x):
        """x: (B, n_cpgs) → mu, logvar: (B, latent_dim)"""
        h      = self.encoder(x)
        mu     = self.z_mean(h)
        logvar = self.z_var(h)
        return mu, logvar

    def decode(self, z):
        """z: (B, latent_dim) → recon: (B, n_cpgs)"""
        return self.decoder(z)

    def get_latent_z(self, x):
        mu, logvar = self.encode(x)
        return self.sample_z(mu, logvar)

    def forward(self, x):
        """返回 (recon, mu, logvar)"""
        mu, logvar = self.encode(x)
        z          = self.sample_z(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, loss_func, epoch: int,
             kl_warm_up: int = 0, beta: float = 1.0):
    """VAE 损失 = 重建损失 + KL 散度（带 warm-up 和 beta 权重）。

    量级对齐：
    - recon_loss: loss_func(reduction='sum') / batch_size
      → 对 batch 维度平均，对特征维度累加
    - kl_loss: mean(sum(..., dim=1))
      → 对 batch 维度平均，对特征维度累加
    两者量级一致，避免 KL 压制重建损失导致后验坍塌。

    参数
    ----
    loss_func  : nn.MSELoss(reduction='sum')
    kl_warm_up : 前 N epoch KL 权重从 0 线性增加到 beta
    beta       : KL 权重（beta-VAE，建议 0.1~1.0）
    """
    if not isinstance(recon, list):
        recon = [recon]
    batch_size = x.size(0)
    # sum 后除以 batch_size：对 batch 平均，对特征维度累加
    recon_loss = sum(loss_func(r, x) for r in recon) / batch_size

    # 特征维度求和，batch 维度求平均
    kl_loss = torch.mean(
        0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=1)
    )
    kl_loss = kl_loss * beta
    if epoch < kl_warm_up:
        kl_loss = kl_loss * float(np.clip(epoch / kl_warm_up, 0.0, 1.0))

    return recon_loss + kl_loss, recon_loss, kl_loss
