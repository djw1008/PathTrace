"""ContrastivePathwayTransformer.py

PathwayTransformer + 对比学习（双样本对比分支）

架构：
  1. PathwayTokenizer  : 每条通路 CpG -> MLP -> latent token  (B, N, d)
  2. LearnedPosEnc     : 可学习位置编码
  3. TransformerEncoder: 标准多层自注意力（Pre-LN）
  4. TokenAggregator   : weighted softmax 汇聚 (B, N, d) -> (B, d)
  5. Predictor         : MLP -> 归一化年龄标量 (B, 1)
  6. DifferencePredictor: 预测年龄差（对比学习分支）
"""

from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_pathway_encoder(n_cpgs: int, hidden: List[int], latent_dim: int) -> nn.Sequential:
    dims = [n_cpgs] + hidden
    layers: list = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        nn.init.xavier_uniform_(lin.weight)
        layers.extend([lin, nn.ReLU()])
    out = nn.Linear(dims[-1], latent_dim)
    nn.init.xavier_uniform_(out.weight)
    layers.append(out)
    return nn.Sequential(*layers)


def _mlp(dims: List[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class PathwayTokenizer(nn.Module):
    """每条通路的 CpG 子集 -> 一个 latent token。"""

    def __init__(self, pathway_cpg_indices: List[List[int]],
                 latent_dim: int = 32,
                 hidden: Optional[List[int]] = None):
        super().__init__()
        if hidden is None:
            hidden = [64, 64]
        self._cpg_indices = pathway_cpg_indices
        self.n_pathways = len(pathway_cpg_indices)
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleList(
            [_build_pathway_encoder(len(idx), hidden, latent_dim)
             for idx in pathway_cpg_indices]
        )

    def _cache_idx_tensors(self, device: torch.device):
        if not hasattr(self, '_cpg_idx_tensors') or self._cpg_idx_tensors[0].device != device:
            self._cpg_idx_tensors = [
                torch.tensor(idx, dtype=torch.long, device=device)
                for idx in self._cpg_indices
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, total_cpgs) -> (B, N, latent_dim)"""
        out = [enc(x[:, idx_t]) for idx_t, enc in
               zip(self._cpg_idx_tensors, self.encoders)]
        return torch.stack(out, dim=1)


class LearnedPosEnc(nn.Module):
    def __init__(self, n_pathways: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(n_pathways, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.unsqueeze(0)


class TokenAggregator(nn.Module):
    """(B, N, d) -> (B, d)，weighted softmax 模式支持通路可解释性。"""

    def __init__(self, n_pathways: int, d_model: int, mode: str = 'weighted'):
        super().__init__()
        self.mode = mode
        if mode == 'weighted':
            self.weights = nn.Parameter(torch.ones(n_pathways))
        elif mode == 'attention':
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
            self.attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        elif mode != 'mean':
            raise ValueError(f"aggregator_mode 须为 mean/weighted/attention，得到 '{mode}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'mean':
            return x.mean(dim=1)
        elif self.mode == 'weighted':
            w = F.softmax(self.weights, dim=0)
            return (x * w[None, :, None]).sum(dim=1)
        else:
            cls = self.cls.expand(x.size(0), -1, -1)
            out, _ = self.attn(cls, x, x)
            return self.norm(out.squeeze(1))

    def get_pathway_weights(self) -> Optional[torch.Tensor]:
        if self.mode == 'weighted':
            return F.softmax(self.weights.detach(), dim=0)
        return None


class ContrastivePathwayTransformer(nn.Module):
    """PathwayTransformer + 对比学习（双样本对比分支）

    同时学习：
    1. 绝对年龄预测（监督学习）
    2. 年龄差异预测（对比学习）
    """

    def __init__(
        self,
        pathway_cpg_indices: List[List[int]],
        latent_dim: int = 16,
        hidden_topo: Optional[List[int]] = None,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        aggregator_mode: str = 'attention',
        predictor_hidden: Optional[List[int]] = None,
        diff_predictor_hidden: Optional[List[int]] = None,
        use_pos_enc: bool = True,
        token_dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_topo is None:
            hidden_topo = [64, 64]
        if predictor_hidden is None:
            predictor_hidden = [64, 32]
        if diff_predictor_hidden is None:
            diff_predictor_hidden = [32, 16]

        n_pathways = len(pathway_cpg_indices)
        d_model = latent_dim

        # 编码器（与PathwayTransformer相同）
        self.tokenizer = PathwayTokenizer(pathway_cpg_indices, latent_dim, hidden_topo)
        self.use_pos_enc = use_pos_enc
        self.pos_enc = LearnedPosEnc(n_pathways, d_model) if use_pos_enc else None
        self.token_dropout = token_dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        self.aggregator = TokenAggregator(n_pathways, d_model, aggregator_mode)

        # 绝对年龄预测头（回归模式）
        pred_dims = [d_model] + predictor_hidden + [1]
        self.age_predictor = _mlp(pred_dims, dropout=dropout)

        # 对比学习分支：年龄差异预测头
        # 输入：[enc1, enc2, |enc1-enc2|] -> 3 * d_model
        diff_input_dim = d_model * 3
        diff_dims = [diff_input_dim] + diff_predictor_hidden + [1]
        self.diff_predictor = _mlp(diff_dims, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间
        
        Args:
            x: (B, total_cpgs)
        Returns:
            (B, latent_dim)
        """
        self.tokenizer._cache_idx_tensors(x.device)
        tokens = self.tokenizer(x)   # (B, N, d)
        if self.use_pos_enc and self.pos_enc is not None:
            tokens = self.pos_enc(tokens)
        # Token Dropout：训练时随机将整条通路的 token 置零，迫使模型分散依赖
        if self.training and self.token_dropout > 0:
            mask = torch.rand(tokens.size(0), tokens.size(1), 1,
                              device=tokens.device) > self.token_dropout
            tokens = tokens * mask / (1 - self.token_dropout)
        hs = self.transformer(tokens)               # (B, N, d)
        return self.aggregator(hs)                  # (B, d)

    def forward(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None, mode: str = 'predict') -> torch.Tensor:
        if mode == 'predict':
            # 绝对年龄预测（推理 & 训练右脚）
            enc = self.encode(x)
            return self.age_predictor(enc)

        elif mode == 'contrastive':
            # 对比学习（训练左脚）
            enc1 = self.encode(x)
            enc2 = self.encode(x2)
            diff_feat = torch.cat([enc1, enc2, enc1 - enc2], dim=1)
            age_diff = self.diff_predictor(diff_feat)
            return age_diff

        else:
            raise ValueError(f"mode 须为 predict/contrastive，得到 '{mode}'")

    def get_pathway_importance(self) -> Optional[torch.Tensor]:
        return self.aggregator.get_pathway_weights()
