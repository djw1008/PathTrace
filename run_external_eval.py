"""run_external_eval.py

对 PathTrace / 其消融变体在外部测试集上做跨 GSE 和跨组织评估。
逻辑参考 MAPLE-main/jupyter/run_both_models_on_gse.ipynb 中 TRACE 的评估部分。

用法：
    python run_external_eval.py \
        --checkpoint ./checkpoints/abl_none/checkpoints/best_model.pt \
        --beta_path /path/to/beta_values_all.csv \
        --meta_path /path/to/meta_data_v7.csv \
        --output_dir ./external_eval_results
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from models.ContrastivePathwayTransformer import ContrastivePathwayTransformer, TokenAggregator


def load_model_from_checkpoint(checkpoint_path, device):
    """加载训练好的 PathTrace / 消融模型。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ablation = ckpt.get('ablation', 'none')

    if ablation == 'flat_mlp':
        # Flat MLP: state_dict 是一个 nn.Sequential
        from run_ablations import _build_flat_mlp
        # 从 checkpoint 里恢复结构
        config = ckpt.get('config', {})
        hidden_topo = config.get('flat_mlp_hidden', config.get('hidden_topo', [2048, 1024, 512]))
        feature_size = len(ckpt['cpg_names'])
        flat_dims = [feature_size] + hidden_topo + [1]
        model = _build_flat_mlp(flat_dims, dropout=config.get('dropout', 0.0)).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt, ablation

    # Pathway 模型
    config = ckpt.get('config', {
        'latent_dim': 32,
        'hidden_topo': [64, 64],
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'aggregator_mode': 'attention',
        'predictor_hidden': [64, 32],
        'diff_predictor_hidden': [32, 16],
        'use_pos_enc': False,
    })
    pathway_cpg_idx = ckpt['pathway_cpg_idx']
    model = ContrastivePathwayTransformer(
        pathway_cpg_indices=pathway_cpg_idx,
        latent_dim=config['latent_dim'],
        hidden_topo=config.get('hidden_topo', [64, 64]),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 3),
        dim_feedforward=config.get('dim_feedforward', 128),
        dropout=config.get('dropout', 0.1),
        aggregator_mode=config.get('aggregator_mode', 'attention'),
        predictor_hidden=config.get('predictor_hidden', [64, 32]),
        diff_predictor_hidden=config.get('diff_predictor_hidden', [32, 16]),
        use_pos_enc=config.get('use_pos_enc', False),
    )

    if ablation == 'no_transformer':
        model.transformer = nn.Identity()
        model.aggregator = TokenAggregator(len(pathway_cpg_idx), config['latent_dim'], mode='mean')
        logging.info('[消融] 推理时重建 no_transformer 模型')

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, ckpt, ablation


def run_inference(model, df_beta, cpg_names, cpg_means, x_train_knn, age_norm,
                  device, ablation='none', batch_size=256):
    """对 df_beta 做推理，返回预测年龄和样本 ID。"""
    df_beta_aligned = df_beta.reindex(cpg_names)
    df_beta_aligned = df_beta_aligned.apply(pd.to_numeric, errors='coerce')
    X_test_raw = df_beta_aligned.T.values.astype(np.float32)
    sample_ids = df_beta_aligned.columns.tolist()

    # 缺失值填充
    if cpg_means is not None:
        X_test_filled = X_test_raw.copy()
        nan_mask = np.isnan(X_test_filled)
        X_test_filled = np.where(nan_mask, cpg_means, X_test_filled).astype(np.float32)
        logging.info('使用 checkpoint cpg_means 填充，NaN 数: %d', nan_mask.sum())
    elif x_train_knn is not None:
        from sklearn.impute import KNNImputer
        X_combined = np.vstack([x_train_knn, X_test_raw])
        knn_imp = KNNImputer(n_neighbors=min(5, len(x_train_knn)))
        X_combined_filled = knn_imp.fit_transform(X_combined)
        X_test_filled = X_combined_filled[len(x_train_knn):].astype(np.float32)
        logging.info('使用 KNN 填充')
    else:
        raise RuntimeError('checkpoint 中没有 cpg_means 或 x_train_knn')

    preds = []
    with torch.no_grad():
        for i in range(0, X_test_filled.shape[0], batch_size):
            end = min(i + batch_size, X_test_filled.shape[0])
            batch = torch.from_numpy(X_test_filled[i:end]).to(device)
            if ablation == 'flat_mlp':
                pred = model(batch)
            else:
                pred = model(batch)
            preds.append(pred.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0).flatten() * age_norm
    return y_pred, sample_ids


def map_chip_type(platform):
    if pd.isna(platform):
        return 'Unknown'
    p = str(platform).upper()
    if '935' in p:
        return '935K'
    elif 'EPIC' in p or '850' in p:
        return '850K'
    else:
        return '450K'


def evaluate_per_gse(results_df, output_dir):
    """按 GSE 计算 MedAE、RMSE、Pearson R（仅 control 样本）。"""
    if 'project_id' not in results_df.columns:
        raise KeyError('Meta 中缺少 project_id 列')

    df_ctrl = results_df[results_df.get('sample_type', '') == 'control'].copy()
    if len(df_ctrl) == 0:
        logging.warning('未找到 control 样本，将使用全部样本计算')
        df_ctrl = results_df.copy()

    gse_stats = []
    for gse, group in df_ctrl.groupby('project_id'):
        valid = group.dropna(subset=['true_age', 'predicted_age'])
        if len(valid) == 0:
            continue
        error = valid['predicted_age'] - valid['true_age']
        gse_stats.append({
            'GSE': gse,
            'n_samples': len(valid),
            'MedAE': error.abs().median(),
            'MAE': error.abs().mean(),
            'RMSE': np.sqrt((error ** 2).mean()),
            'Pearson_r': valid['true_age'].corr(valid['predicted_age']),
        })

    stats_df = pd.DataFrame(gse_stats).sort_values('MedAE')
    stats_path = os.path.join(output_dir, 'gse_mae_comparison.csv')
    stats_df.to_csv(stats_path, index=False)
    logging.info('\n按 GSE 统计（已保存 %s）:\n%s', stats_path, stats_df.to_string(index=False))

    overall_mean_of_medians = stats_df['MedAE'].mean()
    overall_median = df_ctrl['abs_error'].median()
    logging.info('总体 MedAE (Mean of per-GSE Medians): %.3f', overall_mean_of_medians)
    logging.info('总体 MedAE (Overall Median):          %.3f', overall_median)
    return stats_df


def evaluate_per_tissue(results_df, output_dir, min_samples=10):
    """按组织类型计算 MedAE（仅 control + 450K 样本）。"""
    if 'tissue' not in results_df.columns:
        logging.warning('Meta 缺少 tissue 列，跳过组织分析')
        return None

    mask_ctrl = results_df.get('sample_type', '') == 'control'
    mask_450k = results_df.get('chip_type', '') == '450K'
    df_tissue = results_df[mask_ctrl & mask_450k].copy()

    if len(df_tissue) == 0:
        logging.warning('未找到 control + 450K 样本，跳过组织分析')
        return None

    logging.info('跨组织分析样本: %d (control + 450k only)', len(df_tissue))

    tissue_stats = []
    for tissue, group in df_tissue.groupby('tissue'):
        valid = group.dropna(subset=['true_age', 'predicted_age'])
        if len(valid) < min_samples:
            continue
        error = valid['predicted_age'] - valid['true_age']
        tissue_stats.append({
            'Tissue': tissue,
            'n_samples': len(valid),
            'MedAE': error.abs().median(),
            'MAE': error.abs().mean(),
            'Pearson_r': valid['true_age'].corr(valid['predicted_age']),
        })

    tissue_df = pd.DataFrame(tissue_stats).sort_values('MedAE')
    tissue_path = os.path.join(output_dir, 'tissue_mae_comparison.csv')
    tissue_df.to_csv(tissue_path, index=False)
    logging.info('\n按组织统计（已保存 %s）:\n%s', tissue_path, tissue_df.to_string(index=False))
    return tissue_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='PathTrace checkpoint 路径')
    p.add_argument('--beta_path', type=str, required=True,
                   help='外部测试集 beta 矩阵 CSV，行为 CpG，列为 sample_id')
    p.add_argument('--meta_path', type=str, required=True,
                   help='共享 Meta CSV，包含 sample_id, age, project_id, sample_type, tissue, platform')
    p.add_argument('--output_dir', type=str, default='./external_eval_results')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--min_tissue_samples', type=int, default=10)
    p.add_argument('--exclude_gse', type=str, default='',
                   help='要排除的 GSE，逗号分隔，如 GSE109042,GSE111223')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger('external_eval')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('使用设备: %s', device)

    # 加载模型
    logger.info('加载 checkpoint: %s', args.checkpoint)
    model, ckpt, ablation = load_model_from_checkpoint(args.checkpoint, device)
    cpg_names = ckpt['cpg_names']
    cpg_means = ckpt.get('cpg_means', None)
    x_train_knn = ckpt.get('x_train_knn', None)
    age_norm = ckpt.get('age_norm', 100.0)
    logger.info('checkpoint ablation: %s, age_norm: %.1f, CpG 数: %d',
                ablation, age_norm, len(cpg_names))

    # 加载 Meta
    logger.info('加载 Meta: %s', args.meta_path)
    df_meta = pd.read_csv(args.meta_path, sep=',')
    id_col = None
    for col in ['sample_id', 'SampleID', 'ID', 'id', 'sample']:
        if col in df_meta.columns:
            id_col = col
            break
    if id_col is None and df_meta.index.name is not None:
        id_col = df_meta.index.name
    if id_col and df_meta.index.name != id_col:
        df_meta = df_meta.set_index(id_col)
    if df_meta.index.name is None:
        df_meta.index.name = 'sample_id'
    df_meta.index = df_meta.index.astype(str)

    if 'age' in df_meta.columns:
        df_meta['age'] = pd.to_numeric(df_meta['age'], errors='coerce')

    # 加载 Beta
    logger.info('加载 Beta: %s', args.beta_path)
    df_beta = pd.read_csv(args.beta_path, sep=',', index_col=0)
    df_beta.columns = df_beta.columns.astype(str)
    logger.info('Beta 维度 (CpG x sample): %s', df_beta.shape)

    common_samples = df_meta.index.intersection(df_beta.columns)
    df_beta = df_beta.loc[:, common_samples]
    df_meta = df_meta.loc[common_samples, :].copy()
    logger.info('共同样本数: %d', len(common_samples))

    # 推理
    y_pred, sample_ids = run_inference(
        model, df_beta, cpg_names, cpg_means, x_train_knn,
        age_norm, device, ablation=ablation, batch_size=args.batch_size
    )

    # 合并结果
    results_df = df_meta.copy().reset_index()
    results_df.rename(columns={results_df.columns[0]: 'sample_id'}, inplace=True)
    results_df['predicted_age'] = y_pred
    if 'age' in results_df.columns:
        results_df['true_age'] = pd.to_numeric(results_df['age'], errors='coerce')
    elif 'true_age' not in results_df.columns:
        raise RuntimeError('Meta 中未找到 age 列')

    results_df['error'] = results_df['predicted_age'] - results_df['true_age']
    results_df['abs_error'] = results_df['error'].abs()

    if 'platform' in results_df.columns:
        results_df['chip_type'] = results_df['platform'].apply(map_chip_type)
    else:
        results_df['chip_type'] = 'Unknown'

    # 动态排除 GSE
    if args.exclude_gse:
        exclude_list = [x.strip() for x in args.exclude_gse.split(',') if x.strip()]
        if 'project_id' in results_df.columns:
            n_before = len(results_df)
            results_df = results_df[~results_df['project_id'].isin(exclude_list)].copy()
            logger.info('排除 GSE %s，样本数从 %d 减少到 %d', exclude_list, n_before, len(results_df))

    # 保存合并结果
    pred_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(pred_path, index=False)
    logger.info('预测结果已保存: %s', pred_path)

    # 评估
    evaluate_per_gse(results_df, args.output_dir)
    evaluate_per_tissue(results_df, args.output_dir, min_samples=args.min_tissue_samples)

    logger.info('外部评估完成，结果目录: %s', args.output_dir)


if __name__ == '__main__':
    main()
