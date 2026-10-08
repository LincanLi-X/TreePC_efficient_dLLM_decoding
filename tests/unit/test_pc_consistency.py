import torch

from treepc.data.onpolicy import _support_from_logits, _support_matrix_from_log_probs
from treepc.posterior.consistency import posterior_consistency_kl


def test_pc_kl_is_zero_for_identical_distribution_with_tail() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    ids = torch.tensor([[0, 1]])
    teacher_support = log_probs.gather(-1, ids)
    tail = 1.0 - teacher_support.exp().sum(-1)
    loss, per_token = posterior_consistency_kl(logits, ids, teacher_support, tail)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(per_token, torch.zeros_like(per_token), atol=1e-6, rtol=0)


def test_pc_kl_penalizes_mismatched_student() -> None:
    teacher_logits = torch.tensor([[3.0, 1.0, 0.0, -1.0]])
    student_logits = -teacher_logits
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    ids = torch.tensor([[0, 1]])
    support = teacher_log_probs.gather(-1, ids)
    tail = 1.0 - support.exp().sum(-1)
    loss, _ = posterior_consistency_kl(student_logits, ids, support, tail)
    assert loss > 0.5


def test_onpolicy_support_preserves_topk_union_and_other_mass() -> None:
    base = torch.tensor([4.0, 3.0, 0.0, -1.0, -2.0])
    target = torch.tensor([-2.0, 0.0, 4.0, 3.0, -1.0])
    ids, mask, base_values, target_values, base_other, target_other = _support_from_logits(
        base, target, torch.tensor(4), top_k=2
    )
    assert set(ids[mask].tolist()) == {0, 1, 2, 3, 4}
    torch.testing.assert_close(base_values[mask].exp().sum() + base_other.exp(), torch.tensor(1.0))
    torch.testing.assert_close(target_values[mask].exp().sum() + target_other.exp(), torch.tensor(1.0))


def test_batched_onpolicy_support_matches_scalar_construction() -> None:
    base = torch.tensor([[4.0, 3.0, 0.0, -1.0], [0.0, 2.0, 3.0, -1.0]])
    target = torch.tensor([[0.0, -1.0, 4.0, 3.0], [3.0, 2.0, -1.0, 0.0]])
    base_log = torch.log_softmax(base, -1)
    target_log = torch.log_softmax(target, -1)
    values = _support_matrix_from_log_probs(
        base_log,
        target_log,
        torch.topk(base, 2, dim=-1).indices,
        torch.topk(target, 2, dim=-1).indices,
        torch.tensor([1, 3]),
        2,
    )
    for row in range(2):
        expected = _support_from_logits(base[row], target[row], torch.tensor([1, 3])[row], 2)
        for batched, scalar in zip(values, expected):
            torch.testing.assert_close(batched[row], scalar)
