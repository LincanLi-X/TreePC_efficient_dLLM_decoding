from __future__ import annotations

import torch
from torch import Tensor

from treepc.posterior.consistency import posterior_consistency_kl


def posterior_metric_vectors(
    student_logits: Tensor,
    teacher_topk_ids: Tensor,
    teacher_topk_log_probs: Tensor,
    teacher_tail_mass: Tensor,
    *,
    recall_k: int = 16,
) -> dict[str, Tensor]:
    """Return per-token RQ1 metrics against a Top-K+OTHER Teacher target."""
    if recall_k <= 0:
        raise ValueError("recall_k must be positive")
    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [tokens, vocab]")
    if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
        raise ValueError("Teacher Top-K ids and log probabilities must have the same shape")
    if teacher_topk_ids.shape[0] != student_logits.shape[0]:
        raise ValueError("Student and Teacher token counts differ")

    logits = student_logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    _, marginal_kl = posterior_consistency_kl(
        logits,
        teacher_topk_ids,
        teacher_topk_log_probs,
        teacher_tail_mass,
    )
    teacher_top1 = teacher_topk_ids[:, 0]
    student_top1 = logits.argmax(dim=-1)
    student_top1_confidence = probabilities.max(dim=-1).values
    width = min(recall_k, logits.shape[-1])
    student_topk = torch.topk(logits, width, dim=-1).indices
    return {
        "marginal_kl": marginal_kl,
        "top1_agreement": student_top1.eq(teacher_top1).float(),
        "teacher_token_recall": student_topk.eq(teacher_top1[:, None]).any(dim=-1).float(),
        "student_entropy": -(probabilities * log_probs).sum(dim=-1),
        "student_top1_confidence": student_top1_confidence,
        "teacher_top1_probability": probabilities.gather(
            -1, teacher_top1[:, None]
        ).squeeze(-1),
    }


def expected_calibration_error(
    confidence: Tensor, correctness: Tensor, *, bins: int = 15
) -> float:
    """ECE where correctness means agreement with the Teacher top-1 token."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    confidence = confidence.float().reshape(-1)
    correctness = correctness.float().reshape(-1)
    if confidence.numel() != correctness.numel() or confidence.numel() == 0:
        raise ValueError("confidence and correctness must be non-empty and aligned")
    edges = torch.linspace(0.0, 1.0, bins + 1, device=confidence.device)
    result = torch.zeros((), dtype=torch.float32, device=confidence.device)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = confidence.ge(lower) & confidence.le(upper)
        if index:
            selected &= confidence.gt(lower)
        if selected.any():
            weight = selected.float().mean()
            gap = correctness[selected].mean() - confidence[selected].mean()
            result += weight * gap.abs()
    return float(result.item())
