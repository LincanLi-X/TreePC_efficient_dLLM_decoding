import torch

from treepc.dream.alignment import align_dream_hidden, align_dream_logits


def test_dream_alignment_shifts_logits_and_hidden_identically() -> None:
    values = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    expected = torch.cat([values[:, :1], values[:, :-1]], dim=1)
    assert torch.equal(align_dream_logits(values), expected)
    assert torch.equal(align_dream_hidden(values), expected)
