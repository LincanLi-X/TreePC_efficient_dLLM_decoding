from __future__ import annotations

import torch
from torch import Tensor


def posterior_consistency_kl(
    student_logits: Tensor,
    teacher_topk_ids: Tensor,
    teacher_topk_log_probs: Tensor,
    teacher_tail_mass: Tensor,
) -> tuple[Tensor, Tensor]:
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    student_support = student_log_probs.gather(-1, teacher_topk_ids)
    student_tail = (1.0 - student_support.exp().sum(dim=-1)).clamp_min(1e-12)
    teacher_tail = teacher_tail_mass.float().clamp_min(1e-12)
    support_probabilities = teacher_topk_log_probs.float().exp()
    support_kl = support_probabilities * (
        teacher_topk_log_probs.float() - student_support
    )
    tail_kl = teacher_tail * (teacher_tail.log() - student_tail.log())
    per_token = support_kl.sum(dim=-1) + tail_kl
    return per_token.mean(), per_token
