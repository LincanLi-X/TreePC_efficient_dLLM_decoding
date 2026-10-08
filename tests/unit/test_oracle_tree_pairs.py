import torch

from treepc.data.cache_dataset import CorrectionCacheDataset
from treepc.training.correction_trainer import (
    select_learned_tree_pairs,
    select_oracle_tree_pairs,
)


class FixedDependencyHead(torch.nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del hidden, positions, timestep
        return self.weights


def _record() -> dict:
    nodes, samples, width = 4, 3, 2
    dependency = torch.tensor(
        [
            [0.0, 9.0, 1.0, 1.0],
            [9.0, 0.0, 8.0, 1.0],
            [1.0, 8.0, 0.0, 7.0],
            [1.0, 1.0, 7.0, 0.0],
        ]
    )
    shape = (nodes, samples, nodes, width)
    other_shape = (nodes, samples, nodes)
    return {
        "state_token_ids": torch.zeros(20, dtype=torch.long),
        "candidate_positions": torch.arange(nodes),
        "candidate_confidence": torch.tensor([0.6, 0.7, 0.9, 0.5]),
        "symmetric_dependency": dependency,
        "predicted_dependency": torch.zeros_like(dependency),
        "directed_dependency": dependency,
        "directed_dependency_samples": dependency[:, None, :].expand(-1, samples, -1),
        "sampled_parent_tokens": torch.arange(nodes)[:, None].expand(-1, samples),
        "aligned_hidden": torch.randn(nodes, 5),
        "timestep": 0.5,
        "support_ids": torch.zeros(shape, dtype=torch.long),
        "support_mask": torch.ones(shape, dtype=torch.bool),
        "base_support_log_probs": torch.full(shape, -1.0),
        "conditional_support_log_probs": torch.full(shape, -1.0),
        "base_other_log_probs": torch.full(other_shape, -1.0),
        "conditional_other_log_probs": torch.full(other_shape, -1.0),
    }


def test_oracle_stage_uses_only_oriented_teacher_mst_pairs() -> None:
    record = _record()
    selected = select_oracle_tree_pairs([record])

    assert selected == {0: {(2, 1), (1, 0), (2, 3)}}
    oracle_dataset = CorrectionCacheDataset([record], identity_fraction=0.0, selected_pairs=selected)
    all_pair_dataset = CorrectionCacheDataset([record], identity_fraction=0.0)

    assert len(oracle_dataset) == (4 - 1) * 3
    assert len(all_pair_dataset) == 4 * (4 - 1) * 3
    assert {(item.parent, item.child) for item in oracle_dataset.indices} == selected[0]
    assert all(int(oracle_dataset[index]["sequence_length"]) == 20 for index in range(9))

    with_identity = CorrectionCacheDataset([record], identity_fraction=0.25, selected_pairs=selected)
    identity_pairs = {(item.parent, item.child) for item in with_identity.indices if item.identity}
    assert identity_pairs
    assert identity_pairs.isdisjoint(selected[0])


def test_learned_tree_adaptation_can_select_a_different_pair_distribution() -> None:
    record = _record()
    learned_weights = torch.tensor(
        [
            [0.0, 1.0, 9.0, 1.0],
            [1.0, 0.0, 8.0, 1.0],
            [9.0, 8.0, 0.0, 7.0],
            [1.0, 1.0, 7.0, 0.0],
        ]
    )

    oracle = select_oracle_tree_pairs([record])
    learned = select_learned_tree_pairs(FixedDependencyHead(learned_weights), [record], torch.device("cpu"))

    assert oracle[0] == {(2, 1), (1, 0), (2, 3)}
    assert learned[0] == {(2, 0), (2, 1), (2, 3)}
    assert learned != oracle
    torch.testing.assert_close(record["predicted_dependency"], learned_weights)
    dataset = CorrectionCacheDataset([record], identity_fraction=0, selected_pairs=learned)
    first = dataset.indices[0]
    assert dataset[0]["dependency_weight"] == learned_weights[first.parent, first.child]
