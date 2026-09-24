from __future__ import annotations

import torch
from torch import Tensor


def categorical_kl_from_logits(q_logits: Tensor, p_logits: Tensor) -> Tensor:
    """Compute KL(q || p) over the full vocabulary in float32."""
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    return torch.sum(q_log.exp() * (q_log - p_log), dim=-1).clamp_min(0)


def categorical_nll_from_logits(logits: Tensor, target: Tensor) -> Tensor:
    return -torch.log_softmax(logits.float(), dim=-1).gather(-1, target[..., None]).squeeze(-1)
