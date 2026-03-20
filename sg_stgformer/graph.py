from __future__ import annotations

import torch


def build_physical_adjacency(
    num_joints: int,
    edges: list[list[int]],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32, device=device)
    for source, target in edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    return adjacency


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=-1).clamp(min=1.0)
    inv_sqrt_degree = degree.pow(-0.5)
    return inv_sqrt_degree.unsqueeze(-1) * adjacency * inv_sqrt_degree.unsqueeze(-2)
