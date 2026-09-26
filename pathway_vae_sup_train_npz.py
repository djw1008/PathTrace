"""pathway_vae_sup_train_npz.py

有监督版本：为每个 Reactome 通路训练带年龄预测头的 PathwayVAESup。
使用 npz 数据格式，加载速度更快。
损失 = 重建损失 + KL散度 + alpha * 年龄回归损失。

用法：
    python pathway_vae_sup_train_npz.py
"""

import os
import sys
import gc
import copy
import time
import logging
import argparse
from datetime import datetime
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.PathwayVAESup import PathwayVAESup, vae_sup_loss
from utils.pathway_utils import (
    load_pathway_cpg_map, get_pathway_name_map, print_pathway_stats
)
from utils.file_utils import FileUtils
from utils.dataload_utils import geo_npz_Dataset_train


# ─────────────────────────────────────────────────────────────────────────────
# Dataset（带年龄标签，从 npz 加载）
# ─────────────────────────────────────────────────────────────────────────────
class NPZPathwayAgeDataset(Dataset):
    """从 npz 加载的通路 CpG 子矩阵 + 年龄标签 Dataset。"""
    def __init__(self, X_subset: np.ndarray, ages: np.ndarray, training: bool = False):
        """
        Args:
            X_subset: (N, n_cpgs_in_pathway) 该通路的 CpG 子矩阵
            ages: (N,) 归一化后的年龄
            training: 是否为训练模式，训练时对整数年龄加均匀噪声
        """
        self.X = torch.from_numpy(X_subset.astype(np.float32))
        self.ages = torch.from_numpy(ages.astype(np.float32))
        self.training = training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        age = self.ages[idx]
        if self.training:
            # 连续标签平滑：若原始年龄为整数（归一化后 *100 为整数），加 [0,1) 均匀噪声
            age_raw = age * 100.0
            if torch.abs(age_raw - torch.round(age_raw)) < 1e-3:
                age = age + torch.rand(1).item() / 100.0
        return self.X[idx], age


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载（从 npz 文件）
# ─────────────────────────────────────────────────────────────────────────────
def load_data_from_npz(
    npz_path: str,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """从 npz 文件加载 Beta 矩阵和年龄标签。

    返回
    ----
    X_train    : (N_train, n_cpgs) float32
    X_val      : (N_val, n_cpgs) float32
    ages_norm  : (N_train + N_val,) float32，归一化后的年龄
    cpg_names  : List[str]
    sample_ids : List[str]
    """
    def log(msg):
        (logger.info if logger else print)(msg)

    log(f"[数据] 读取 npz: {npz_path}")
    data_npy = np.load(npz_path, allow_pickle=True)
    
    # 获取 CpG 名称
    cpg_names = list(data_npy['cpgs'])
    log(f"       CpG 位点数: {len(cpg_names)}")
    
    # 加载训练集和验证集
    train_dataset = geo_npz_Dataset_train(file_npy=npz_path, data_type='train')
    val_dataset = geo_npz_Dataset_train(file_npy=npz_path, data_type='val')
    
    log(f"       训练集样本数: {len(train_dataset)}")
    log(f"       验证集样本数: {len(val_dataset)}")
    
    # 提取特征和年龄
    X_train_list, age_train_list = [], []
    for i in range(len(train_dataset)):
        feature, age, _ = train_dataset[i]
        X_train_list.append(feature)
        age_train_list.append(age)
    
    X_val_list, age_val_list = [], []
    for i in range(len(val_dataset)):
        feature, age, _ = val_dataset[i]
        X_val_list.append(feature)
        age_val_list.append(age)
    
    X_train = np.stack(X_train_list).astype(np.float32)
    X_val = np.stack(X_val_list).astype(np.float32)
    ages_train = np.array(age_train_list, dtype=np.float32)
    ages_val = np.array(age_val_list, dtype=np.float32)
    
    # 合并年龄用于后续索引
    ages_all = np.concatenate([ages_train, ages_val])
    
    # 处理缺失值（用 0.5 填充）
    X_train = np.nan_to_num(X_train, nan=0.5)
    X_val = np.nan_to_num(X_val, nan=0.5)
    
    # 裁剪到 [0, 1]
    X_train = np.clip(X_train, 0, 1)
    X_val = np.clip(X_val, 0, 1)
    
    # 归一化年龄（除以 100）
    ages_norm = (ages_all / 100.0).astype(np.float32)
    
    log(f"       年龄范围: {ages_all.min():.1f} ~ {ages_all.max():.1f} 岁")
    log(f"       归一化后年龄范围: {ages_norm.min():.3f} ~ {ages_norm.max():.3f}")
    
    # 获取样本 ID
    sample_ids = []
    for i in range(len(train_dataset)):
        _, _, additional = train_dataset[i]
        sample_ids.append(additional.get('sample_id', f'train_{i}'))
    for i in range(len(val_dataset)):
        _, _, additional = val_dataset[i]
        sample_ids.append(additional.get('sample_id', f'val_{i}'))
    
    return X_train, X_val, ages_norm, cpg_names, sample_ids


# ─────────────────────────────────────────────────────────────────────────────
# 单通路有监督 VAE 训练
# ─────────────────────────────────────────────────────────────────────────────
def train_one_pathway_vae_sup(
    X_train: np.ndarray,
    X_val: np.ndarray,
    ages_train: np.ndarray,
    ages_val: np.ndarray,
    n_cpgs: int,
    args,
    device: torch.device,
) -> Tuple[PathwayVAESup, float]:
    """训练单个有监督通路 VAE，返回 (best_model, best_val_loss)。"""
    hidden_topo = [int(h) for h in args.hidden_topo.split(',')]

    model = PathwayVAESup(
        n_cpgs=n_cpgs,
        latent_dim=args.latent_dim,
        hidden_layer_encoder_topology=hidden_topo,
    ).to(device)

    recon_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    train_loader = DataLoader(
        NPZPathwayAgeDataset(X_train, ages_train, training=True),
        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        NPZPathwayAgeDataset(X_val, ages_val, training=False),
        batch_size=args.batch_size * 4, shuffle=False, drop_last=False, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    best_val_loss = float('inf')
    best_model = copy.deepcopy(model)
    patience_cnt = 0

    for epoch in range(1, args.vae_epochs + 1):
        # ── 训练 ──
        model.train()
        for (xb, age_b) in train_loader:
            xb = xb.to(device)
            age_b = age_b.to(device)
            recon, mu, logvar, age_pred = model(xb)
            loss, _, _, _ = vae_sup_loss(
                recon, xb, mu, logvar, age_pred, age_b,
                recon_fn, epoch, args.kl_warm_up, args.beta_kl, args.alpha_age
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ── 验证 ──
        model.eval()
        total_val = 0.0
        n_val = len(val_loader.dataset)
        with torch.no_grad():
            for (xb, age_b) in val_loader:
                xb = xb.to(device)
                age_b = age_b.to(device)
                recon, mu, logvar, age_pred = model(xb)
                loss, _, _, _ = vae_sup_loss(
                    recon, xb, mu, logvar, age_pred, age_b,
                    recon_fn, epoch, args.kl_warm_up, args.beta_kl, args.alpha_age
                )
                total_val += loss.item() * len(xb)
        avg_val = total_val / n_val
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model = copy.deepcopy(model)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                break

    return best_model, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# 提取潜在表示
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_latent(
    model: PathwayVAESup,
    X: np.ndarray,
    ages: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """提取全样本潜在均值 mu，返回 (N, latent_dim)。"""
    model.eval()
    loader = DataLoader(
        NPZPathwayAgeDataset(X, ages),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    mus = []
    for (xb, _) in loader:
        xb = xb.to(device)
        mu, _ = model.encode(xb)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        free_mem = [torch.cuda.mem_get_info(i)[0]
                    for i in range(torch.cuda.device_count())]
        best_gpu = int(np.argmax(free_mem))
        device = torch.device(f'cuda:{best_gpu}')
        torch.cuda.set_device(best_gpu)
        logger.info(f"使用 GPU {best_gpu}，空闲显存 {free_mem[best_gpu]/1024**3:.2f} GiB")
    else:
        logger.info("未检测到 GPU，使用 CPU 训练")

    # 1. 加载 npz 数据
    X_train, X_val, ages_norm, cpg_names, sample_ids = load_data_from_npz(
        args.data_source,
        logger=logger,
    )
    N_train = len(X_train)
    N_val = len(X_val)
    N_total = N_train + N_val
    
    logger.info(f"训练集: {N_train} 个样本，验证集: {N_val} 个样本")

    # 2. 构建通路→CpG 映射
    logger.info(f"[通路] 正在从 GMT 文件构建映射: {args.gmt_file}")
    pathway_map = load_pathway_cpg_map(
        gmt_path=args.gmt_file,
        cpg_universe=cpg_names,
        min_cpgs=args.min_cpgs,
        max_cpgs=args.max_cpgs,
    )
    name_map = get_pathway_name_map(args.pathway_txt) if args.pathway_txt else {}
    print_pathway_stats(pathway_map, name_map)
    pathway_ids = sorted(pathway_map.keys())
    n_pathways = len(pathway_ids)
    logger.info(f"有效通路数: {n_pathways} 个")

    # 3. 准备输出目录
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    embed_dir = os.path.join(args.out_dir, 'embeddings')
    FileUtils.makedir(ckpt_dir)
    FileUtils.makedir(embed_dir)

    pd.DataFrame([
        {'pathway_id': pid,
         'pathway_name': name_map.get(pid, pid),
         'n_cpgs': len(pathway_map[pid])}
        for pid in pathway_ids
    ]).to_csv(os.path.join(args.out_dir, 'pathway_info.csv'), index=False)

    pd.DataFrame({'sample_id': sample_ids}).to_csv(
        os.path.join(args.out_dir, 'sample_ids.csv'), index=False
    )
    logger.info(f"通路元信息已保存至 {args.out_dir}/pathway_info.csv")

    # 4. 预分配合并潜在矩阵
    all_latents = np.zeros((N_total, n_pathways * args.latent_dim), dtype=np.float32)
    success_ids: List[str] = []

    ages_train = ages_norm[:N_train]
    ages_val = ages_norm[N_train:]

    t_all = time.time()
    for p_idx, pid in enumerate(pathway_ids):
        cpg_list = pathway_map[pid]
        n_cpgs = len(cpg_list)
        pname = name_map.get(pid, pid)[:60]
        t0 = time.time()

        try:
            col_indices = [cpg_names.index(c) for c in cpg_list]
        except ValueError as e:
            logger.warning(f"[{p_idx+1}/{n_pathways}] {pid} 列索引错误: {e}，跳过")
            continue

        X_sub = np.concatenate([X_train, X_val], axis=0)[:, col_indices]
        X_train_sub = X_sub[:N_train]
        X_val_sub = X_sub[N_train:]

        try:
            best_model, val_loss = train_one_pathway_vae_sup(
                X_train_sub, X_val_sub,
                ages_train, ages_val,
                n_cpgs, args, device
            )
        except Exception as e:
            logger.warning(f"[{p_idx+1}/{n_pathways}] {pid} 训练失败: {e}，跳过")
            continue

        if args.save_checkpoints:
            ckpt_path = os.path.join(ckpt_dir, f"{pid}.pt")
            torch.save({
                'state_dict': best_model.state_dict(),
                'cpg_list': cpg_list,
                'n_cpgs': n_cpgs,
                'latent_dim': args.latent_dim,
                'hidden_topo': args.hidden_topo,
            }, ckpt_path)

        mu_all = extract_latent(best_model, X_sub, ages_norm, device,
                                batch_size=args.batch_size * 4)
        col_start = p_idx * args.latent_dim
        col_end = col_start + args.latent_dim
        all_latents[:, col_start:col_end] = mu_all
        success_ids.append(pid)

        elapsed = time.time() - t0
        total_e = time.time() - t_all
        remain = total_e / (p_idx + 1) * (n_pathways - p_idx - 1)
        logger.info(
            f"[{p_idx+1:4d}/{n_pathways}] {pid} | {pname} "
            f"| {n_cpgs} 个CpG | 验证损失={val_loss:.5f} "
            f"| 耗时:{elapsed:.1f}秒 | 剩余≈{remain/60:.1f}分钟"
        )

        del best_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # 5. 保存合并潜在矩阵
    success_indices = [pathway_ids.index(pid) for pid in success_ids]
    valid_cols = np.hstack([
        np.arange(i * args.latent_dim, (i + 1) * args.latent_dim)
        for i in success_indices
    ])
    final_latents = all_latents[:, valid_cols]

    embed_path = os.path.join(args.out_dir, 'pathway_embeddings.npy')
    np.save(embed_path, final_latents)
    logger.info(f"\n合并潜在矩阵已保存: {embed_path}  shape={final_latents.shape}")

    pd.DataFrame({'pathway_id': success_ids}).to_csv(
        os.path.join(args.out_dir, 'success_pathways.csv'), index=False
    )
    logger.info(f"成功训练通路数: {len(success_ids)} / {n_pathways} 个")
    logger.info(f"总耗时: {(time.time()-t_all)/60:.1f} 分钟")
    logger.info("有监督 VAE 训练完成！")


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='有监督通路 VAE 训练（npz 版本）')

    # 数据路径（使用 npz 格式）
    p.add_argument('--data_source', type=str,
                   default='./data/train.npz',
                   help='训练数据 npz 文件路径')
    p.add_argument('--gmt_file', type=str,
                   default='./data/pathways/ReactomePathways.gmt')
    p.add_argument('--pathway_txt', type=str,
                   default='./data/pathways/ReactomePathways.txt')
    p.add_argument('--out_dir', type=str,
                   default='./pretrained_vae')

    # 通路筛选
    p.add_argument('--min_cpgs', type=int, default=50)
    p.add_argument('--max_cpgs', type=int, default=50000)

    # VAE 模型
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--hidden_topo', type=str, default='64,64')

    # 训练超参
    p.add_argument('--vae_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--kl_warm_up', type=int, default=10)
    p.add_argument('--beta_kl', type=float, default=1.0,
                   help='KL 损失权重')
    p.add_argument('--alpha_age', type=float, default=1.0,
                   help='年龄回归损失权重（越大越偏向年龄预测，越小越偏向重建）')
    p.add_argument('--patience', type=int, default=10)

    p.add_argument('--save_checkpoints', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    FileUtils.makedir(os.path.join(args.out_dir, 'logs'))

    log_file = os.path.join(
        args.out_dir, 'logs',
        f'pathway_vae_sup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger('pathway_vae_sup')
    logger.info('=' * 70)
    logger.info('有监督通路 VAE 训练（npz 版本）')
    logger.info(f'  npz 文件    : {args.data_source}')
    logger.info(f'  GMT文件     : {args.gmt_file}')
    logger.info(f'  输出目录    : {args.out_dir}')
    logger.info(f'  潜在维度    : {args.latent_dim}')
    logger.info(f'  隐藏层结构  : {args.hidden_topo}')
    logger.info(f'  训练轮数    : {args.vae_epochs}')
    logger.info(f'  年龄损失权重: {args.alpha_age}')
    logger.info(f'  KL损失权重  : {args.beta_kl}')
    logger.info(f'  最少CpG数   : {args.min_cpgs}')
    logger.info('=' * 70)

    main(args, logger)
