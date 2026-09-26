"""pathway_utils.py

从 Reactome GMT 文件构建 CpG → 通路 映射工具。

GMT 中 gene 列格式为 'GENENAME_cgXXXXXX'，
需提取 '_' 后面的 CpG ID 与 Beta 矩阵列名匹配。
"""

import os
import sys
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_pathway_cpg_map(
    gmt_path: str,
    cpg_universe: List[str],
    min_cpgs: int = 5,
    max_cpgs: int = 2000,
) -> Dict[str, List[str]]:
    """从 GMT 文件构建 通路ID → CpG列表 的映射字典。

    GMT gene 列格式为 'GENE_cgXXXXXX'，提取 '_cg' 之后的 CpG ID。

    参数
    ----
    gmt_path     : ReactomePathways.gmt 文件路径
    cpg_universe : Beta 矩阵实际存在的 CpG 列名列表（过滤不存在的位点）
    min_cpgs     : 通路至少需要覆盖的 CpG 数（过少则跳过，默认 5）
    max_cpgs     : 通路最多保留的 CpG 数（过多内存压力大，默认 2000）

    返回
    ----
    dict: {pathway_id: [cpg1, cpg2, ...]}
    """
    cpg_set = set(cpg_universe)
    pathway_map: Dict[str, List[str]] = {}

    with open(gmt_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            # 列0: 通路名称（human-readable），列1: Reactome ID，列2+: GENE_cgXXX
            pathway_id   = parts[1]          # e.g. R-HSA-164843
            gene_cpg_entries = parts[2:]     # e.g. ['BANF1_cg13588572', ...]

            # 提取 CpG ID：取 '_cg' 最后一次出现后的内容
            cpgs = []
            for entry in gene_cpg_entries:
                # 匹配 _cg 或 _ch 开头的位点 ID
                m = re.search(r'_(cg\w+|ch\.\S+)$', entry)
                if m:
                    cpg_id = m.group(1)
                    if cpg_id in cpg_set:
                        cpgs.append(cpg_id)

            # 去重，保留顺序
            seen = set()
            cpgs_unique = []
            for c in cpgs:
                if c not in seen:
                    seen.add(c)
                    cpgs_unique.append(c)

            if min_cpgs <= len(cpgs_unique) <= max_cpgs:
                pathway_map[pathway_id] = cpgs_unique

    return pathway_map


def get_pathway_name_map(pathway_txt_path: str) -> Dict[str, str]:
    """从 ReactomePathways.txt 构建 ReactomeID → 通路名称 的映射。"""
    name_map: Dict[str, str] = {}
    df = pd.read_csv(pathway_txt_path, sep='\t', header=None,
                     names=['reactome_id', 'pathway_name', 'species'])
    # 只保留人类通路
    human = df[df['reactome_id'].str.contains('HSA', na=False)]
    for _, row in human.iterrows():
        name_map[row['reactome_id']] = row['pathway_name']
    return name_map


def print_pathway_stats(pathway_map: Dict[str, List[str]],
                        name_map: Dict[str, str] = None) -> None:
    """打印通路覆盖统计信息。"""
    sizes = [len(v) for v in pathway_map.values()]
    print(f"有效通路数      : {len(pathway_map)}")
    print(f"CpG 覆盖范围    : {min(sizes)} ~ {max(sizes)} 个/通路")
    print(f"中位数          : {np.median(sizes):.0f} 个/通路")
    print(f"总 CpG 条目数   : {sum(sizes)}")
    if name_map:
        print("\n前 5 个通路示例:")
        for pid, cpgs in list(pathway_map.items())[:5]:
            name = name_map.get(pid, pid)
            print(f"  {pid} | {name[:50]:50s} | {len(cpgs)} CpGs")


# ─────────────────────────────────────────────────────────────────────────────
# 独立验证入口：python utils/pathway_utils.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='验证 CpG 位点提取是否正确')
    parser.add_argument('--gmt_file',    type=str,
        default='./data/pathways/ReactomePathways.gmt',
        help='GMT 文件路径')
    parser.add_argument('--pathway_txt', type=str,
        default='./data/pathways/ReactomePathways.txt',
        help='ReactomePathways.txt 路径')
    parser.add_argument('--beta_file',   type=str,
        default='./data/train_beta.csv',
        help='Beta 矩阵 CSV 路径（只读取列名，不加载数据）')
    parser.add_argument('--min_cpgs',    type=int, default=5)
    parser.add_argument('--max_cpgs',    type=int, default=2000)
    args = parser.parse_args()

    print('=' * 60)
    print('[步骤 1] 读取 Beta 矩阵行索引（CpG 列表），仅读前 2 行，速度极快...')
    # Beta 格式：行=CpG位点，列=样本，index_col=0 读出的 index 就是 CpG 名
    beta_head = pd.read_csv(args.beta_file, index_col=0, nrows=2)
    # 用 nrows=None + usecols=[0] 只读第一列（CpG 名），最快
    beta_cols = pd.read_csv(args.beta_file, usecols=[0], header=0).iloc[:, 0].tolist()
    print(f'   Beta 文件中的 CpG 总数: {len(beta_cols)}')
    print(f'   前 5 个 CpG: {beta_cols[:5]}')
    print(f'   后 5 个 CpG: {beta_cols[-5:]}')
    print(f'   是否以 "cg" 开头: {all(str(c).startswith("cg") for c in beta_cols[:20])}')

    print()
    print('[步骤 2] 读取 GMT 文件，抽查前 3 行原始内容...')
    with open(args.gmt_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.strip().split('\t')
            print(f'   行 {i}: 通路名={parts[0][:40]}  ReactomeID={parts[1]}')
            print(f'          前 3 个 gene_cg 条目: {parts[2:5]}')
            # 演示正则提取
            import re
            for entry in parts[2:5]:
                m = re.search(r'_(cg\w+|ch\.\S+)$', entry)
                extracted = m.group(1) if m else '【未匹配】'
                in_beta   = '✓ 在Beta中' if extracted in set(beta_cols) else '✗ 不在Beta中'
                print(f'          {entry}  →  提取到: {extracted}  {in_beta}')

    print()
    print('[步骤 3] 构建完整通路→CpG 映射...')
    pathway_map = load_pathway_cpg_map(
        gmt_path=args.gmt_file,
        cpg_universe=beta_cols,
        min_cpgs=args.min_cpgs,
        max_cpgs=args.max_cpgs,
    )

    print()
    print('[步骤 4] 统计结果...')
    name_map = get_pathway_name_map(args.pathway_txt)
    print_pathway_stats(pathway_map, name_map)

    print()
    print('[步骤 5] 抽查 3 个通路的 CpG 列表...')
    for pid, cpgs in list(pathway_map.items())[:3]:
        pname = name_map.get(pid, pid)
        print(f'   通路: {pid} | {pname[:50]}')
        print(f'   CpG 数量: {len(cpgs)}')
        print(f'   前 5 个 CpG: {cpgs[:5]}')
        # 验证这些 CpG 确实在 Beta 列名里
        found    = sum(1 for c in cpgs if c in set(beta_cols))
        print(f'   验证: {found}/{len(cpgs)} 个 CpG 确认存在于 Beta 矩阵中')
        print()

    print('[完成] 如果上面 "确认存在" 的数量 == CpG 数量，说明位点提取完全正确！')
    print('=' * 60)
