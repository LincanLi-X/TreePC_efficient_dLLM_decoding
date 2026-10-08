import torch

from treepc.posterior.divergences import categorical_kl_from_logits
from treepc.posterior.topk_support import topk_log_probs_with_tail


def test_full_vocabulary_kl_and_topk_tail_are_well_formed() -> None:
    p = torch.tensor([[1.0, -0.5, 0.2, 2.0]])
    q = torch.tensor([[0.1, 1.5, -0.2, 0.0]])
    assert categorical_kl_from_logits(p, p).item() < 1e-7
    assert categorical_kl_from_logits(q, p).item() > 0
    _, log_probs, tail = topk_log_probs_with_tail(q, 2)
    assert torch.allclose(log_probs.exp().sum(dim=-1) + tail, torch.ones(1), atol=1e-6)
