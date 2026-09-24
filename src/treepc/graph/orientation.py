from __future__ import annotations

import torch
from torch import Tensor


def orient_tree(prim_parent: Tensor, root: int | Tensor) -> tuple[Tensor, Tensor]:
    """Orient an undirected Prim tree away from a requested root."""
    nodes = prim_parent.numel()
    device = prim_parent.device
    adjacency = torch.zeros((nodes, nodes), dtype=torch.bool, device=device)
    child = torch.arange(nodes, device=device)
    valid = prim_parent >= 0
    adjacency[prim_parent[valid], child[valid]] = True
    adjacency[child[valid], prim_parent[valid]] = True
    root_tensor = torch.as_tensor(root, device=device, dtype=torch.long)
    parent = torch.full((nodes,), -2, dtype=torch.long, device=device)
    depth = torch.full((nodes,), -1, dtype=torch.long, device=device)
    visited = torch.zeros(nodes, dtype=torch.bool, device=device)
    frontier = torch.zeros_like(visited)
    frontier[root_tensor] = True
    visited[root_tensor] = True
    parent[root_tensor] = -1
    depth[root_tensor] = 0
    for level in range(1, nodes):
        connected = (adjacency & frontier[:, None]).any(dim=0)
        new_nodes = connected & ~visited
        candidate_parents = adjacency & frontier[:, None]
        selected_parent = candidate_parents.long().argmax(dim=0)
        parent = torch.where(new_nodes, selected_parent, parent)
        depth = torch.where(new_nodes, torch.full_like(depth, level), depth)
        visited |= new_nodes
        frontier = new_nodes
    return parent, depth
