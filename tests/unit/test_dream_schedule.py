import torch

from treepc.dream.adapter import DreamAdapter


def test_commit_budget_matches_official_formula_and_force_finish() -> None:
    masked = torch.ones((1, 32), dtype=torch.bool)
    assert DreamAdapter.compute_commit_budget(masked, torch.tensor(1.0), torch.tensor(0.75), False) == 8
    assert DreamAdapter.compute_commit_budget(masked, torch.tensor(0.25), torch.tensor(0.001), True) == 32
