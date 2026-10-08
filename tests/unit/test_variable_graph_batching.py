import torch
from torch.utils.data import DataLoader

from treepc.data.cache_dataset import DependencyBatchSampler
from treepc.training.onpolicy_pc import RolloutBuffer


def test_mixed_budget_graphs_batch_without_padding():
    sizes = [2, 4, 8, 4, 2, 8, 4]
    data = [{"positions": torch.arange(size), "hidden": torch.zeros(size, 3)} for size in sizes]
    sampler = DependencyBatchSampler(data, 2, shuffle=True, generator=torch.Generator().manual_seed(7))
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(len(sizes)))
    assert len(batches) == len(sampler)
    assert all(len({sizes[i] for i in batch}) == 1 for batch in batches)
    for batch in DataLoader(data, batch_sampler=sampler):
        assert batch["hidden"].ndim == 3


def test_rollout_sampling_is_uniform_by_budget_not_state_count():
    buffer = RolloutBuffer(None, [], {"student_budgets": [4, 8]}, seed=11)
    buffer.records = [{"budget": 4}] * 100 + [{"budget": 8}]
    chosen = [buffer.sample()["budget"] for _ in range(1000)]
    assert 400 < chosen.count(8) < 600
