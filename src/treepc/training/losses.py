from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def dependency_loss(
    prediction: Tensor,
    raw_scores: Tensor,
    target: Tensor,
    *,
    regression_weight: float = 1.0,
    ranking_weight: float = 0.25,
    symmetry_weight: float = 0.01,
) -> tuple[Tensor, dict[str, Tensor]]:
    nodes = prediction.shape[-1]
    upper = torch.triu(torch.ones((nodes, nodes), dtype=torch.bool, device=prediction.device), 1)
    predicted_edges = prediction[:, upper]
    target_edges = target[:, upper]
    regression = F.huber_loss(predicted_edges, target_edges)
    target_difference = target_edges.unsqueeze(-1) - target_edges.unsqueeze(-2)
    prediction_difference = predicted_edges.unsqueeze(-1) - predicted_edges.unsqueeze(-2)
    comparable = target_difference.abs() > 1e-5
    if comparable.any():
        ranking = F.softplus(-target_difference.sign()[comparable] * prediction_difference[comparable]).mean()
    else:
        ranking = prediction.new_zeros(())
    raw_reverse_edges = raw_scores.transpose(-1, -2)[:, upper]
    symmetry = (raw_scores[:, upper] - raw_reverse_edges).abs().mean()
    total = regression_weight * regression + ranking_weight * ranking + symmetry_weight * symmetry
    return total, {"regression": regression, "ranking": ranking, "symmetry": symmetry}


def conditional_kl_loss(
    corrected_support_logits: Tensor,
    base_other_log_prob: Tensor,
    target_support_log_probs: Tensor,
    target_other_log_prob: Tensor,
    support_mask: Tensor,
    effective_correction: Tensor,
    *,
    delta_weight: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    minimum = torch.finfo(corrected_support_logits.dtype).min
    support_logits = corrected_support_logits.masked_fill(~support_mask, minimum)
    all_logits = torch.cat([support_logits, base_other_log_prob.unsqueeze(-1)], dim=-1)
    predicted_log_probs = torch.log_softmax(all_logits, dim=-1)
    target_support = target_support_log_probs.masked_fill(~support_mask, -torch.inf)
    target_all = torch.cat([target_support, target_other_log_prob.unsqueeze(-1)], dim=-1)
    target_probabilities = target_all.exp()
    finite_target = torch.where(torch.isfinite(target_all), target_all, torch.zeros_like(target_all))
    per_item = (target_probabilities * (finite_target - predicted_log_probs)).sum(dim=-1)
    delta_squared_norm = (
        effective_correction.masked_fill(~support_mask, 0.0).square().sum(dim=-1)
    )
    return per_item.mean() + delta_weight * delta_squared_norm.mean(), per_item
