"""Train PathTrace on methylation data stored in NPZ format.

Architecture:
  PathwayTokenizer -> LearnedPosEnc -> TransformerEncoder(自注意力)
  -> TokenAggregator(weighted) -> 两个分支：
     1. MLP -> 年龄（绝对值）
     2. MLP -> 年龄差（对比学习）
"""

import argparse
import gc
import logging
import os
import time
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

from utils.dataload_utils import geo_npz_Dataset_train
from utils.pathway_utils import load_pathway_cpg_map
from models.ContrastivePathwayTransformer import ContrastivePathwayTransformer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.device('cpu')


# =============================================================================
# Dataset adapter
# =============================================================================

class NPZContrastiveDataset(Dataset):
    """Return methylation features and chronological age."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feature, age, additional = self.base_dataset[idx]
        if not isinstance(feature, torch.Tensor):
            feature = torch.from_numpy(feature).float()
        if not isinstance(age, torch.Tensor):
            age = torch.tensor(age, dtype=torch.float32)
        if age.dim() == 0:
            age = age.unsqueeze(0)
        return feature, age


# =============================================================================
# Pathway-index construction
# =============================================================================

def build_pathway_indices_from_npz(gmt_path, cpg_names, min_cpgs=5, max_cpgs=500,
                                   max_pathways=0, logger=None):
    """Build pathway-specific CpG indices from a GMT annotation file."""

    def log(m):
        (logger.info if logger else print)(m)

    log('[通路构建] 解析 GMT 文件...')
    pathway_map = load_pathway_cpg_map(gmt_path, cpg_names,
                                       min_cpgs=min_cpgs, max_cpgs=max_cpgs)
    log(f'  有效通路数: {len(pathway_map)}')

    sorted_ids = sorted(pathway_map, key=lambda k: len(pathway_map[k]), reverse=True)
    selected = sorted_ids if max_pathways <= 0 else sorted_ids[:max_pathways]
    log(f'  保留通路数: {len(selected)}')

    cpg_name2idx = {c: i for i, c in enumerate(cpg_names)}
    pathway_ids, pathway_cpg_idx = [], []
    for pid in selected:
        idx = [cpg_name2idx[c] for c in pathway_map[pid] if c in cpg_name2idx]
        if len(idx) >= min_cpgs:
            pathway_ids.append(pid)
            pathway_cpg_idx.append(idx)

    log(f'  最终通路数: {len(pathway_ids)}')
    if pathway_ids:
        sizes = [len(v) for v in pathway_cpg_idx]
        log(f'  通路大小范围: {min(sizes)} - {max(sizes)} CpG/通路')
    return pathway_ids, pathway_cpg_idx


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, loader, prefix, epoch, logger, age_norm=100.0):
    """Evaluate absolute-age predictions in years."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            age_pred = model(X_batch).cpu().numpy()
            preds.append(age_pred)
            trues.append(y_batch.numpy())

    y_pred = np.vstack(preds).flatten() * age_norm
    y_true = np.vstack(trues).flatten()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_pred) > 1 else 0.0
    medae = float(np.median(np.abs(y_pred - y_true)))
    logger.info(
        'epoch:%4d [%8s]  MAE:%.3f岁  RMSE:%.3f岁  R:%.3f  MedAE:%.3f岁',
        epoch, prefix, mae, rmse, corr, medae
    )
    return mae, rmse, corr, medae, y_pred, y_true


# =============================================================================
# Plotting
# =============================================================================

def plot_scatter(y_true, y_pred, mae, rmse, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, s=15, c='steelblue', edgecolors='none')
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='理想预测')
    ax.set_xlabel('真实年龄（岁）')
    ax.set_ylabel('预测年龄（岁）')
    ax.set_title(f'ContrastivePathwayTransformer Age Prediction\nMAE={mae:.2f}y  RMSE={rmse:.2f}y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pathway_importance(weights, pathway_ids, save_path, top_n=30):
    idx = np.argsort(weights)[::-1][:top_n]
    names = [pathway_ids[i] for i in idx]
    vals = weights[idx]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
    ax.barh(range(len(names)), vals[::-1], color='steelblue', alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xlabel('Importance Weight (Softmax)')
    ax.set_title(f'Top-{top_n} Pathway Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Training
# =============================================================================

def train(args, logger):
    global device

    logger.info('=' * 60)
    logger.info('[数据加载] 从 npz 文件加载数据...')
    logger.info(f'  数据文件: {args.data_source}')

    train_base = geo_npz_Dataset_train(file_npy=args.data_source, data_type='train')
    val_base = geo_npz_Dataset_train(file_npy=args.data_source, data_type='val')

    train_ds = NPZContrastiveDataset(train_base)
    val_ds = NPZContrastiveDataset(val_base)

    data_npy = np.load(args.data_source, allow_pickle=True)
    cpg_names = list(data_npy['cpgs'])
    feature_size = len(cpg_names)

    train_index = data_npy['train_index']
    data_dict = data_npy['data'].item()

    x_train_list = []
    for key in train_index:
        if key in data_dict:
            x_train_list.append(data_dict[key]['feature'])
    x_train = np.array(x_train_list, dtype=np.float32)

    # Training-set CpG means are stored for inference-time imputation.
    cpg_means = np.mean(x_train, axis=0).astype(np.float32)
    logger.info(f'  已计算训练集CpG均值，用于推理缺失值填充')

    knn_max_samples = 2000
    if len(x_train) > knn_max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(x_train), size=knn_max_samples, replace=False)
        x_train_knn = x_train[indices].astype(np.float32)
    else:
        x_train_knn = x_train.astype(np.float32)
    logger.info(f'  保存训练集子集用于KNN插补: {len(x_train_knn)}样本')

    logger.info(f'  CpG 位点数: {feature_size}')
    logger.info(f'  训练集样本数: {len(train_ds)}')
    logger.info(f'  验证集样本数: {len(val_ds)}')
    logger.info('=' * 60)

    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    pathway_ids, pathway_cpg_idx = build_pathway_indices_from_npz(
        args.gmt_file, cpg_names,
        min_cpgs=args.min_cpgs, max_cpgs=args.max_cpgs,
        max_pathways=args.max_pathways, logger=logger)

    if not pathway_cpg_idx:
        raise RuntimeError('未找到有效通路，请检查 --gmt_file 和 CpG 名称。')

    hidden_topo = [int(h) for h in args.hidden_topo.split(',')]
    pred_hidden = [int(h) for h in args.pred_hidden.split(',')]
    diff_hidden = [int(h) for h in args.diff_predictor_hidden.split(',')]

    model = ContrastivePathwayTransformer(
        pathway_cpg_indices=pathway_cpg_idx,
        latent_dim=args.latent_dim,
        hidden_topo=hidden_topo,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        aggregator_mode=args.aggregator_mode,
        predictor_hidden=pred_hidden,
        diff_predictor_hidden=diff_hidden,
        use_pos_enc=args.use_pos_enc,
    ).to(device)

    if torch.cuda.device_count() > 1:
        gpu_id = device.index if device.type == 'cuda' else 0
        logger.info('多卡并行，主卡: cuda:%d', gpu_id)
        model = nn.DataParallel(model, device_ids=[gpu_id])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_p = sum(p.numel() for p in trainable_params)
    frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info('可训练参数量: %d，已冻结参数量: %d', total_p, frozen_p)
    logger.info('通路数: %d，latent_dim: %d，Transformer 层数: %d',
                len(pathway_ids), args.latent_dim, args.num_layers)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # The two objectives update a shared encoder through separate optimizers.
    encoder_params = list(raw_model.tokenizer.parameters()) + \
                     list(raw_model.transformer.parameters()) + \
                     list(raw_model.aggregator.parameters())
    if raw_model.use_pos_enc and raw_model.pos_enc is not None:
        encoder_params += list(raw_model.pos_enc.parameters())

    contrast_params = encoder_params + list(raw_model.diff_predictor.parameters())
    contrast_optimizer = optim.AdamW(contrast_params, lr=args.lr, weight_decay=1e-4)

    predictor_params = encoder_params + list(raw_model.age_predictor.parameters())
    predictor_optimizer = optim.AdamW(predictor_params, lr=args.lr * 0.5, weight_decay=1e-4)

    scheduler_c = optim.lr_scheduler.CosineAnnealingLR(
        contrast_optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)
    scheduler_p = optim.lr_scheduler.CosineAnnealingLR(
        predictor_optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01 * 0.5)

    AGE_NORM = 100.0
    criterion_age = nn.HuberLoss(delta=args.huber_delta / AGE_NORM)
    criterion_diff = nn.HuberLoss(delta=args.huber_delta / AGE_NORM)

    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        logger.info('启用 Automatic Mixed Precision (AMP) 训练')

    best_mae, best_epoch = float('inf'), 0
    t0 = time.time()
    logger.info('=' * 60)
    logger.info('开始训练（1:1全量交替训练），共 %d epoch', args.num_epochs)
    logger.info('=' * 60)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        raw_model_ref = model.module if isinstance(model, nn.DataParallel) else model
        total_diff_loss = 0.0
        total_age_loss = 0.0
        steps = 0

        # Stage 1: pairwise age-difference learning.
        for X_batch, y_batch in train_ld:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) / AGE_NORM
            batch_size = X_batch.size(0)

            perm = torch.randperm(batch_size, device=device)
            X_batch_2 = X_batch[perm]
            y_batch_2 = y_batch[perm]

            with autocast(device_type='cuda', enabled=args.use_amp):
                enc1 = raw_model_ref.encode(X_batch)
                enc2 = raw_model_ref.encode(X_batch_2)
                diff_feat = torch.cat([enc1, enc2, enc1 - enc2], dim=1)
                age_diff_pred = raw_model_ref.diff_predictor(diff_feat)

                true_diff = y_batch.squeeze(-1) - y_batch_2.squeeze(-1)
                loss_diff = criterion_diff(age_diff_pred.squeeze(-1), true_diff)

            contrast_optimizer.zero_grad()
            if scaler:
                scaler.scale(loss_diff).backward()
                scaler.unscale_(contrast_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(contrast_optimizer)
                scaler.update()
            else:
                loss_diff.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                contrast_optimizer.step()

            total_diff_loss += loss_diff.item()
            steps += 1

        scheduler_c.step()
        avg_diff_loss = total_diff_loss / max(steps, 1)

        # Stage 2: absolute-age regression.
        for X_batch, y_batch in train_ld:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) / AGE_NORM

            with autocast(device_type='cuda', enabled=args.use_amp):
                age_pred = model(X_batch)
                loss_age = criterion_age(age_pred.squeeze(-1), y_batch.squeeze(-1))

            predictor_optimizer.zero_grad()
            if scaler:
                scaler.scale(loss_age).backward()
                scaler.unscale_(predictor_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(predictor_optimizer)
                scaler.update()
            else:
                loss_age.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                predictor_optimizer.step()

            total_age_loss += loss_age.item()

        scheduler_p.step()
        avg_age_loss = total_age_loss / max(steps, 1)

        elapsed = time.time() - t0
        remaining = elapsed / epoch * (args.num_epochs - epoch)

        if epoch % args.val_interval == 0:
            model.eval()
            evaluate(model, train_ld, '训练集', epoch, logger, AGE_NORM)
            mae, rmse, corr, medae, y_pred, y_true = evaluate(
                model, val_ld, '验证集', epoch, logger, AGE_NORM)

            improved = mae < best_mae
            logger.info('[%4d/%d] age:%.5f diff:%.5f lr_c:%.2e lr_p:%.2e | MAE:%.4f [%s] | 剩余:%.1fmin',
                        epoch, args.num_epochs, avg_age_loss, avg_diff_loss,
                        contrast_optimizer.param_groups[0]['lr'],
                        predictor_optimizer.param_groups[0]['lr'],
                        mae, '新最优' if improved else '未改善', remaining / 60)

            if improved:
                best_mae, best_epoch = mae, epoch
                raw = model.module if isinstance(model, nn.DataParallel) else model
                if args.save_model:
                    ckpt = {
                        'epoch': epoch,
                        'model_state_dict': raw.state_dict(),
                        'pathway_ids': pathway_ids,
                        'pathway_cpg_idx': pathway_cpg_idx,
                        'cpg_names': cpg_names,
                        'cpg_means': cpg_means,
                        'x_train_knn': x_train_knn,
                        'age_norm': AGE_NORM,
                        'config': {
                            'latent_dim': args.latent_dim,
                            'hidden_topo': hidden_topo,
                            'nhead': args.nhead,
                            'num_layers': args.num_layers,
                            'dim_feedforward': args.dim_feedforward,
                            'dropout': args.dropout,
                            'aggregator_mode': args.aggregator_mode,
                            'predictor_hidden': pred_hidden,
                            'diff_predictor_hidden': diff_hidden,
                            'use_pos_enc': args.use_pos_enc,
                        }
                    }
                    ckpt_path = os.path.join(args.path_save, 'checkpoints', 'best_model.pt')
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(ckpt, ckpt_path)
                    logger.info('  模型已保存 -> %s', ckpt_path)
                plot_scatter(y_true, y_pred, mae, rmse,
                             os.path.join(args.path_save, 'best_scatter.png'))
                logger.info('  散点图已保存')
                pw = raw.get_pathway_importance()
                if pw is not None:
                    pw_np = pw.cpu().numpy()
                    plot_pathway_importance(
                        pw_np, pathway_ids,
                        os.path.join(args.path_save, 'pathway_importance.png'))
                    pd.DataFrame({'pathway_id': pathway_ids, 'importance': pw_np}) \
                        .sort_values('importance', ascending=False) \
                        .to_csv(os.path.join(args.path_save, 'pathway_importance.csv'), index=False)
                    logger.info('  通路重要性已保存')

            logger.info('  最优：epoch %d  MAE %.4f 岁', best_epoch, best_mae)
            if epoch - best_epoch >= args.patience and best_epoch > 0:
                logger.info('触发早停。')
                break
            model.train()
        else:
            logger.info('[%4d/%d] age:%.5f diff:%.5f lr_c:%.2e lr_p:%.2e | 剩余:%.1fmin',
                        epoch, args.num_epochs, avg_age_loss, avg_diff_loss,
                        contrast_optimizer.param_groups[0]['lr'],
                        predictor_optimizer.param_groups[0]['lr'], remaining / 60)

    logger.info('训练完成。最优MAE:%.4f岁(epoch%d) 耗时%.1fmin',
                best_mae, best_epoch, (time.time() - t0) / 60)

    return os.path.join(args.path_save, 'checkpoints', 'best_model.pt') if best_epoch > 0 else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_source', type=str,
                   default='./data/train.npz')
    p.add_argument('--gmt_file', type=str,
                   default='./data/pathways/ReactomePathways.gmt')
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--min_cpgs', type=int, default=50)
    p.add_argument('--max_cpgs', type=int, default=50000)
    p.add_argument('--max_pathways', type=int, default=0)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--hidden_topo', type=str, default='128,128')
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--dim_feedforward', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0)
    p.add_argument('--aggregator_mode', type=str, default='attention', choices=['mean', 'weighted', 'attention'])
    p.add_argument('--pred_hidden', type=str, default='64,32')
    p.add_argument('--use_pos_enc', action='store_true', default=False)
    p.add_argument('--diff_predictor_hidden', type=str, default='32,16')
    p.add_argument('--contrast_weight', type=float, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--num_epochs', type=int, default=500)
    p.add_argument('--val_interval', type=int, default=10)
    p.add_argument('--patience', type=int, default=80)
    p.add_argument('--huber_delta', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_model', action='store_true', default=True)
    p.add_argument('--path_save', type=str, default='./checkpoints/contrastive_transformer')
    p.add_argument('--n_pairs_per_batch', type=int, default=5)
    p.add_argument('--predictor_interval', type=int, default=3)
    p.add_argument('--use_amp', action='store_true', default=False)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.path_save, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.path_save, 'logs'), exist_ok=True)
    log_file = os.path.join(args.path_save, 'logs', f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    logger = logging.getLogger('ContrastivePathwayTransformer')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        free_mem = []
        for i in range(torch.cuda.device_count()):
            try:
                f, t = torch.cuda.mem_get_info(i)
                free_mem.append(f)
                logger.info('GPU %d: 空闲%.2fGiB/总计%.2fGiB', i, f / 1024 ** 3, t / 1024 ** 3)
            except RuntimeError:
                free_mem.append(0)
        best_gpu = int(np.argmax(free_mem))
        device = torch.device(f'cuda:{best_gpu}')
        torch.cuda.set_device(best_gpu)
        logger.info('自动选择 GPU %d', best_gpu)
    else:
        device = torch.device('cpu')
    logger.info('使用设备: %s', device)

    logger.info('=' * 60)
    for k, v in vars(args).items():
        logger.info('   %-25s: %s', k, v)
    logger.info('=' * 60)
    t_start = time.time()
    best_checkpoint = train(args, logger)
    logger.info('训练完成！总耗时%.1f分钟', (time.time() - t_start) / 60)
