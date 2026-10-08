import torch

from treepc.dream.adapter import max_token_probability, sample_tokens


def test_root_uses_max_probability_when_entropy_ranking_disagrees() -> None:
    probabilities = torch.tensor(
        [
            [0.55, 0.225, 0.225],
            [0.50, 0.49, 0.01],
        ]
    )
    logits = probabilities.log()

    entropy_confidence, _ = sample_tokens(logits, alg="entropy")
    root_confidence = max_token_probability(logits)

    assert entropy_confidence.argmax().item() == 1
    assert root_confidence.argmax().item() == 0
    torch.testing.assert_close(root_confidence, torch.tensor([0.55, 0.50]))
