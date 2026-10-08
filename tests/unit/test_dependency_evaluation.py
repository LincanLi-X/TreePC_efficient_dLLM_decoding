from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from treepc.training.dependency_trainer import (
    _kendall_tau_b,
    _spearman,
    evaluate_dependency_head,
)


class FixedDependency(torch.nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        return_raw_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del positions, timestep
        prediction = self.weights.expand(hidden.shape[0], -1, -1)
        raw_scores = prediction.clone()
        return (prediction, raw_scores) if return_raw_scores else prediction


def test_rank_correlations_are_tie_aware() -> None:
    increasing = torch.tensor([1.0, 1.0, 2.0, 3.0])
    decreasing = torch.tensor([3.0, 2.0, 1.0, 1.0])

    torch.testing.assert_close(_spearman(increasing, increasing), torch.tensor(1.0))
    assert _kendall_tau_b(increasing, increasing) == 1.0
    assert _kendall_tau_b(torch.arange(4), torch.arange(3, -1, -1)) == -1.0
    assert _kendall_tau_b(increasing, decreasing) < 0.0


def test_dependency_evaluation_reports_uniform_tree_metrics() -> None:
    target = torch.tensor(
        [
            [0.0, 0.9, 0.2, 0.1],
            [0.9, 0.0, 0.8, 0.3],
            [0.2, 0.8, 0.0, 0.7],
            [0.1, 0.3, 0.7, 0.0],
        ]
    )
    batch = {
        "hidden": torch.eye(4),
        "positions": torch.tensor([1, 4, 8, 13]),
        "timestep": torch.tensor(0.5),
        "target": target,
        "record_key": torch.tensor(42),
    }
    metrics = evaluate_dependency_head(
        FixedDependency(target),
        DataLoader([batch], batch_size=1, shuffle=False),
        torch.device("cpu"),
    )

    assert metrics["mae"] == 0.0
    assert math.isclose(metrics["spearman"], 1.0, rel_tol=1e-6)
    assert math.isclose(metrics["kendall_tau_b"], 1.0, rel_tol=1e-6)
    assert set(metrics["trees"]) == {
        "learned",
        "local",
        "hidden_cosine",
        "random",
        "oracle",
    }
    for tree_metrics in metrics["trees"].values():
        assert set(tree_metrics) == {"mst_edge_overlap", "oracle_weight_capture"}
        assert all(math.isfinite(value) for value in tree_metrics.values())
    assert metrics["trees"]["learned"]["mst_edge_overlap"] == 1.0
    assert metrics["trees"]["learned"]["oracle_weight_capture"] == 1.0
    assert metrics["trees"]["oracle"]["mst_edge_overlap"] == 1.0
    assert metrics["trees"]["oracle"]["oracle_weight_capture"] == 1.0
