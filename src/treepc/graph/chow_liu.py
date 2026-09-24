from __future__ import annotations

import torch
from torch import Tensor


def maximum_spanning_tree(weights: Tensor, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Batched deterministic Prim MST; returns Prim parent and selected edge weight."""
    squeeze = weights.ndim == 2
    if squeeze:
        weights = weights.unsqueeze(0)
    batch, nodes, _ = weights.shape
    if valid_mask is None:
        valid_mask = torch.ones((batch, nodes), dtype=torch.bool, device=weights.device)
    elif valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    parent = torch.full((batch, nodes), -1, dtype=torch.long, device=weights.device)
    edge_weight = torch.zeros((batch, nodes), dtype=weights.dtype, device=weights.device)
    if nodes <= 1:
        return (parent[0], edge_weight[0]) if squeeze else (parent, edge_weight)
    root = valid_mask.long().argmax(dim=-1)
    selected = torch.zeros_like(valid_mask).scatter(1, root[:, None], True)
    rows = torch.arange(batch, device=weights.device)
    best_weight = weights[rows, root]
    best_parent = root[:, None].expand(-1, nodes).clone()
    best_weight = best_weight.masked_fill(~valid_mask | selected, -torch.inf)
    for _ in range(nodes - 1):
        next_node = best_weight.argmax(dim=-1)
        active = valid_mask.sum(dim=-1) > selected.sum(dim=-1)
        chosen_weight = best_weight[rows, next_node]
        parent[rows, next_node] = torch.where(active, best_parent[rows, next_node], parent[rows, next_node])
        edge_weight[rows, next_node] = torch.where(active, chosen_weight, edge_weight[rows, next_node])
        selected[rows, next_node] |= active
        candidate = weights[rows, next_node]
        improve = (candidate > best_weight) & ~selected & valid_mask & active[:, None]
        best_weight = torch.where(improve, candidate, best_weight)
        best_parent = torch.where(improve, next_node[:, None], best_parent)
        best_weight = best_weight.masked_fill(selected | ~valid_mask, -torch.inf)
    parent = parent.masked_fill(~valid_mask, -1)
    return (parent[0], edge_weight[0]) if squeeze else (parent, edge_weight)


def tree_weight(weights: Tensor, parent: Tensor) -> Tensor:
    nodes = torch.arange(parent.numel(), device=parent.device)
    valid = parent >= 0
    return weights[parent[valid], nodes[valid]].sum()
