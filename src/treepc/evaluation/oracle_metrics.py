from __future__ import annotations

from typing import Any

import torch

from treepc.graph.chow_liu import maximum_spanning_tree, tree_weight
from treepc.graph.orientation import orient_tree


def _captured_weight(weights: torch.Tensor, prim_parent: torch.Tensor) -> float:
    return float(tree_weight(weights, prim_parent).item())


def evaluate_cached_record(record: dict[str, Any], seed: int = 2026) -> dict[str, Any]:
    weights = record["symmetric_dependency"].float()
    nodes = weights.shape[0]
    oracle_parent, _ = maximum_spanning_tree(weights)
    root = int(record["candidate_confidence"].argmax().item())
    oriented_parent, depth = orient_tree(oracle_parent, root)
    generator = torch.Generator().manual_seed(seed + int(record["step_index"]))
    random_weights = torch.rand((nodes, nodes), generator=generator)
    random_weights = 0.5 * (random_weights + random_weights.T)
    random_weights.fill_diagonal_(0)
    random_parent, _ = maximum_spanning_tree(random_weights)
    local_parent = torch.arange(nodes, dtype=torch.long) - 1
    local_parent[0] = -1
    oracle_weight = _captured_weight(weights, oracle_parent)
    random_weight = _captured_weight(weights, random_parent)
    local_weight = _captured_weight(weights, local_parent)
    children = (oriented_parent >= 0).nonzero(as_tuple=False).flatten()
    parents = oriented_parent[children]
    base_nll = record["base_target_nll"][children].float()
    directed_kl = record["directed_dependency"][parents, children].float()
    final_targets = record["final_teacher_tokens"][record["candidate_positions"]][children]
    if record["sampled_parent_tokens"].ndim == 2:
        conditional_nll = record["conditional_target_nll"][parents, :, children].float()
        conditional_top1 = record["conditional_topk_ids"][parents, :, children, 0]
        conditional_tail = record["conditional_tail_mass"][parents, :, children]
        conditional_targets = final_targets[:, None].expand_as(conditional_top1)
    else:
        conditional_nll = record["conditional_target_nll"][parents, children].float()
        conditional_top1 = record["conditional_topk_ids"][parents, children, 0]
        conditional_tail = record["conditional_tail_mass"][parents, children]
        conditional_targets = final_targets
    base_tokens = record["base_sampled_tokens"][children]
    return {
        "sample_id": record["sample_id"],
        "dataset_index": int(record["dataset_index"]),
        "teacher_step": int(record["step_index"]),
        "mask_ratio": float(record["mask_ratio"]),
        "candidate_size": nodes,
        "tree_depth": int(depth.max().item()),
        "oracle_dependency_sum": oracle_weight,
        "random_dependency_sum": random_weight,
        "local_dependency_sum": local_weight,
        "oracle_over_random_gain": oracle_weight - random_weight,
        "oracle_over_local_gain": oracle_weight - local_weight,
        "independent_conditional_kl": float(directed_kl.mean().item()),
        "oracle_conditional_kl": 0.0,
        "base_teacher_token_nll": float(base_nll.mean().item()),
        "conditional_teacher_token_nll": float(conditional_nll.mean().item()),
        "base_teacher_top1_agreement": float(base_tokens.eq(final_targets).float().mean().item()),
        "conditional_teacher_top1_agreement": float(
            conditional_top1.eq(conditional_targets).float().mean().item()
        ),
        "mean_tail_mass": float(conditional_tail.mean().item()),
    }
