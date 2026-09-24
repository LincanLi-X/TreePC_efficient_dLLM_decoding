from __future__ import annotations

import torch
from torch import Tensor


def align_dream_logits(raw_logits: Tensor) -> Tensor:
    """Map Dream's shifted predictions to their target token positions."""
    return _shift_right(raw_logits)


def align_dream_hidden(last_hidden: Tensor) -> Tensor:
    """Apply the same target-position mapping used for Dream logits."""
    return _shift_right(last_hidden)


def _shift_right(values: Tensor) -> Tensor:
    if values.ndim < 2 or values.shape[1] == 0:
        raise ValueError("Dream alignment expects a non-empty sequence dimension")
    return values[:, :1] if values.shape[1] == 1 else torch.cat([values[:, :1], values[:, :-1]], dim=1)
