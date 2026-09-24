from __future__ import annotations

import torch
from torch import Tensor


def topk_log_probs_with_tail(logits: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(log_probs, min(k, logits.shape[-1]), dim=-1)
    tail = (1.0 - values.exp().sum(dim=-1)).clamp_min(0)
    return indices, values, tail


def support_mass(logits: Tensor, token_ids: Tensor) -> Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities.gather(-1, token_ids).sum(dim=-1)
