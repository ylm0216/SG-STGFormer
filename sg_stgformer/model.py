from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_physical_adjacency, normalize_adjacency


def build_sinusoidal_encoding(max_len: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(max_len, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * ff_multiplier
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveHybridGraph(nn.Module):
    def __init__(self, dim: int, num_joints: int, edges: list[list[int]]) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.alpha_logit = nn.Parameter(torch.zeros(1))
        self.register_buffer("physical_adj", build_physical_adjacency(num_joints, edges), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint_summary = x.mean(dim=1)
        query = self.q_proj(joint_summary)
        key = self.k_proj(joint_summary)
        adaptive = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1)),
            dim=-1,
        )
        alpha = torch.sigmoid(self.alpha_logit)
        physical = self.physical_adj.unsqueeze(0).expand(x.size(0), -1, -1)
        hybrid = alpha * physical + (1.0 - alpha) * adaptive
        return hybrid, alpha


class GraphConvolution(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        num_joints = adjacency.size(-1)
        identity = torch.eye(num_joints, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        normalized = normalize_adjacency(adjacency + identity)
        aggregated = torch.einsum("bij,btjd->btid", normalized, x)
        return self.proj(aggregated)


class SpatialGraphTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, ff_multiplier: int) -> None:
        super().__init__()
        self.graph_conv = GraphConvolution(dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_logit = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_multiplier, dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, frames, joints, dim = x.shape
        gcn_out = self.graph_conv(x, adjacency)

        flattened = x.reshape(batch_size * frames, joints, dim)
        attn_out, _ = self.self_attention(flattened, flattened, flattened, need_weights=False)
        attn_out = attn_out.reshape(batch_size, frames, joints, dim)

        gate = torch.sigmoid(self.gate_logit)
        fused = gcn_out + gate * attn_out
        x = self.norm1(x + self.dropout(fused))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, gate


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, ff_multiplier: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_multiplier, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.self_attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.space_to_time = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.time_to_space = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.space_norm = nn.LayerNorm(dim)
        self.time_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )

    def forward(self, spatial_tokens: torch.Tensor, temporal_tokens: torch.Tensor) -> torch.Tensor:
        spatial_context, _ = self.space_to_time(
            spatial_tokens,
            temporal_tokens,
            temporal_tokens,
            need_weights=False,
        )
        temporal_context, _ = self.time_to_space(
            temporal_tokens,
            spatial_tokens,
            spatial_tokens,
            need_weights=False,
        )

        spatial_tokens = self.space_norm(spatial_tokens + self.dropout(spatial_context))
        temporal_tokens = self.time_norm(temporal_tokens + self.dropout(temporal_context))

        pooled = torch.cat([spatial_tokens.mean(dim=1), temporal_tokens.mean(dim=1)], dim=-1)
        return self.out_proj(pooled)


class SGSTGFormer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        data_cfg = config["data"]
        model_cfg = config["model"]

        input_dim = data_cfg["input_dim"]
        num_classes = data_cfg["num_classes"]
        hidden_dim = model_cfg["hidden_dim"]
        num_joints = data_cfg["num_joints"]
        max_len = model_cfg["max_len"]

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.graph_builder = AdaptiveHybridGraph(hidden_dim, num_joints, config["physical_edges"])
        self.spatial_blocks = nn.ModuleList(
            [
                SpatialGraphTransformerBlock(
                    hidden_dim,
                    model_cfg["num_heads"],
                    model_cfg["dropout"],
                    model_cfg["ff_multiplier"],
                )
                for _ in range(model_cfg["spatial_layers"])
            ]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    hidden_dim,
                    model_cfg["num_heads"],
                    model_cfg["dropout"],
                    model_cfg["ff_multiplier"],
                )
                for _ in range(model_cfg["temporal_layers"])
            ]
        )
        self.register_buffer(
            "sinusoidal_position",
            build_sinusoidal_encoding(max_len, hidden_dim).unsqueeze(0),
            persistent=False,
        )
        self.temporal_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        self.fusion = CrossAttentionFusion(hidden_dim, model_cfg["num_heads"], model_cfg["dropout"])
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(model_cfg["dropout"]),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.embedding_head = nn.Linear(hidden_dim, model_cfg["embedding_dim"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_norm(self.input_proj(x))
        adjacency, alpha = self.graph_builder(x)

        gates = []
        for block in self.spatial_blocks:
            x, gate = block(x, adjacency)
            gates.append(gate)

        spatial_tokens = x.mean(dim=1)
        temporal_tokens = x.mean(dim=2)
        temporal_tokens = (
            temporal_tokens
            + self.sinusoidal_position[:, : temporal_tokens.size(1)]
            + self.temporal_embedding[:, : temporal_tokens.size(1)]
        )

        for block in self.temporal_blocks:
            temporal_tokens = block(temporal_tokens)

        fused = self.shared(self.fusion(spatial_tokens, temporal_tokens))
        score = self.regressor(fused).squeeze(-1)
        logits = self.classifier(fused)
        embedding = F.normalize(self.embedding_head(fused), dim=-1)

        return {
            "score": score,
            "logits": logits,
            "embedding": embedding,
            "hybrid_adjacency": adjacency,
            "alpha": alpha,
            "spatial_gates": torch.stack(gates),
        }
