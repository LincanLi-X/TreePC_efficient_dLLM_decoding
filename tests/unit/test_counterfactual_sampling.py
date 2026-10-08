from types import SimpleNamespace

import torch

from treepc.data.cache_dataset import CorrectionCacheDataset
from treepc.data.cache_schema import validate_counterfactual_bundle
from treepc.data.onpolicy import label_pc_onpolicy_record


class FakeTeacher:
    """Tiny deterministic teacher used to test counterfactual sampling on CPU."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.mask_token_id = 99
        self.forward_calls = 0
        self.base_logits = torch.tensor(
            [
                [2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 2.0, 1.0, -1.0, -2.0],
                [1.0, 0.0, 2.0, -1.0, -2.0],
            ]
        )

    def forward_state(self, state, need_hidden: bool = False):
        self.forward_calls += 1
        logits = self.base_logits.clone()
        revealed = state.input_ids[0, :3].ne(self.mask_token_id).nonzero().flatten()
        if revealed.numel():
            parent = int(revealed[0])
            parent_token = int(state.input_ids[0, parent])
            for child in range(3):
                if child != parent:
                    logits[child, (parent_token + child) % logits.shape[-1]] += 1.5
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        return SimpleNamespace(aligned_logits=logits.unsqueeze(0), aligned_hidden=hidden)


def _record() -> dict:
    return {
        "sample_id": "tiny/0",
        "dataset_index": 0,
        "dataset": "tiny",
        "steps": 4,
        "calibration_fold": "train",
        "state_token_ids": torch.full((3,), 99, dtype=torch.long),
        "generation_mask": torch.ones(3, dtype=torch.bool),
        "prompt_length": 0,
        "step_index": 1,
        "timestep": 0.75,
        "candidate_positions": torch.arange(3),
        "candidate_confidence": torch.ones(3),
        "candidate_confidence_type": "max_token_probability",
        "aligned_hidden": torch.zeros(3, 4),
        "pc_base_logits": torch.randn(3, 5),
        # Deliberately outside the Teacher vocabulary, proving it is not reused.
        "pc_base_tokens": torch.full((3,), 9, dtype=torch.long),
        "final_pc_tokens": torch.zeros(3, dtype=torch.long),
        "seed": 1234,
    }


def test_onpolicy_counterfactual_uses_three_teacher_posterior_samples() -> None:
    teacher = FakeTeacher()
    record = _record()
    labeled = label_pc_onpolicy_record(teacher, record, top_k=2, parent_samples=3)

    assert labeled["parent_sampling_distribution"] == "teacher_posterior"
    assert labeled["parent_samples"] == 3
    assert labeled["candidate_confidence_type"] == "max_token_probability"
    torch.testing.assert_close(
        labeled["candidate_confidence"],
        torch.softmax(record["pc_base_logits"], dim=-1).amax(dim=-1),
    )
    assert labeled["sampled_parent_tokens"].shape == (3, 3)
    assert labeled["sampled_parent_tokens"].max().item() < 5
    assert torch.all(labeled["sampled_parent_tokens"] != 9)
    assert teacher.forward_calls == 1 + 3 * 3

    teacher_probabilities = torch.softmax(teacher.base_logits, dim=-1)
    expected_probabilities = teacher_probabilities.gather(1, labeled["sampled_parent_tokens"])
    torch.testing.assert_close(labeled["parent_probabilities"], expected_probabilities)
    torch.testing.assert_close(
        labeled["directed_dependency"],
        labeled["directed_dependency_samples"].mean(dim=1),
    )

    validate_counterfactual_bundle({"schema": "treepc.counterfactual.v1", "records": [labeled]})
    labeled["predicted_dependency"] = torch.zeros(3, 3)
    correction = CorrectionCacheDataset([labeled], identity_fraction=0.0)
    assert len(correction) == 3 * 3 * 2
    assert {int(correction[index]["parent_token"]) for index in range(len(correction))} == set(
        labeled["sampled_parent_tokens"].flatten().tolist()
    )

    repeated = label_pc_onpolicy_record(FakeTeacher(), _record(), top_k=2, parent_samples=3)
    torch.testing.assert_close(repeated["sampled_parent_tokens"], labeled["sampled_parent_tokens"])
