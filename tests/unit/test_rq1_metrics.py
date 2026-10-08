import pytest
import torch

from treepc.evaluation.rq1 import expected_calibration_error, posterior_metric_vectors


def test_rq1_metrics_are_exact_for_an_identical_teacher_and_student() -> None:
    logits = torch.tensor([[4.0, 2.0, 1.0, -1.0], [0.0, 3.0, 1.0, -2.0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    ids = torch.topk(logits, 2, dim=-1).indices
    teacher_support = log_probs.gather(-1, ids)
    teacher_tail = 1.0 - teacher_support.exp().sum(dim=-1)

    metrics = posterior_metric_vectors(logits, ids, teacher_support, teacher_tail, recall_k=2)

    torch.testing.assert_close(metrics["marginal_kl"], torch.zeros(2), atol=1e-6, rtol=0)
    assert metrics["top1_agreement"].tolist() == [1.0, 1.0]
    assert metrics["teacher_token_recall"].tolist() == [1.0, 1.0]


def test_rq1_recall_and_calibration_detect_a_wrong_confident_student() -> None:
    student = torch.tensor([[-4.0, 5.0, 1.0]])
    teacher = torch.tensor([[5.0, 0.0, -2.0]])
    teacher_log_probs = torch.log_softmax(teacher, dim=-1)
    ids = torch.tensor([[0, 2]])
    support = teacher_log_probs.gather(-1, ids)
    tail = 1.0 - support.exp().sum(dim=-1)

    metrics = posterior_metric_vectors(student, ids, support, tail, recall_k=1)

    assert metrics["marginal_kl"].item() > 1.0
    assert metrics["top1_agreement"].item() == 0.0
    assert metrics["teacher_token_recall"].item() == 0.0
    assert (
        expected_calibration_error(metrics["student_top1_confidence"], metrics["top1_agreement"], bins=5)
        > 0.9
    )


def test_rq1_metric_validation_rejects_invalid_recall_k() -> None:
    with pytest.raises(ValueError, match="recall_k"):
        posterior_metric_vectors(
            torch.zeros(1, 3),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1),
            torch.ones(1),
            recall_k=0,
        )
