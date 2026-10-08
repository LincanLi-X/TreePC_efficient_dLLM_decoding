import math

import torch

from treepc.training.losses import conditional_kl_loss


def test_conditional_loss_regularizes_effective_correction_squared_l2() -> None:
    support_probabilities = torch.tensor([[0.4, 0.3, 0.0], [0.2, 0.5, 0.0]])
    other_probabilities = torch.tensor([0.3, 0.3])
    support_mask = torch.tensor([[True, True, False], [True, True, False]])
    support_log_probs = support_probabilities.clamp_min(1e-30).log()
    other_log_probs = other_probabilities.log()
    effective_correction = torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 100.0]])

    unregularized, per_item = conditional_kl_loss(
        support_log_probs,
        other_log_probs,
        support_log_probs,
        other_log_probs,
        support_mask,
        effective_correction,
        delta_weight=0.0,
    )
    regularized, _ = conditional_kl_loss(
        support_log_probs,
        other_log_probs,
        support_log_probs,
        other_log_probs,
        support_mask,
        effective_correction,
        delta_weight=0.2,
    )

    expected_mean_squared_l2 = torch.tensor(((1.0**2 + 2.0**2) + (3.0**2 + 4.0**2)) / 2)
    torch.testing.assert_close(per_item, torch.zeros_like(per_item), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(unregularized, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        regularized - unregularized,
        0.2 * expected_mean_squared_l2,
        atol=1e-6,
        rtol=0.0,
    )


def test_conditional_loss_penalty_has_gradients_for_full_effective_correction() -> None:
    corrected_logits = torch.tensor([[math.log(0.4), math.log(0.3)]])
    other_log_prob = torch.tensor([math.log(0.3)])
    support_mask = torch.ones_like(corrected_logits, dtype=torch.bool)
    effective_correction = torch.tensor([[0.25, -0.5]], requires_grad=True)

    loss, _ = conditional_kl_loss(
        corrected_logits,
        other_log_prob,
        corrected_logits,
        other_log_prob,
        support_mask,
        effective_correction,
        delta_weight=0.4,
    )
    loss.backward()

    torch.testing.assert_close(effective_correction.grad, 0.8 * effective_correction.detach())
