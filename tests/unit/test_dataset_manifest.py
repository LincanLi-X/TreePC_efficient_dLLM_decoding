import json
from pathlib import Path

import pytest
import yaml

from treepc.data.datasets import build_large_scale_manifest, build_subset_manifest


@pytest.fixture
def benchmark_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "data"
    for name, size in (("gsm8k", 1319), ("humaneval", 164)):
        dataset_dir = data_root / name
        dataset_dir.mkdir(parents=True)
        rows = []
        for index in range(size):
            if name == "gsm8k":
                rows.append({"sample_id": str(index), "question": f"q{index}", "answer": str(index)})
            else:
                rows.append(
                    {
                        "sample_id": f"HumanEval/{index}",
                        "prompt": f"def f{index}():\n",
                        "test": "",
                        "entry_point": f"f{index}",
                    }
                )
        (dataset_dir / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    monkeypatch.setenv("TREEPC_DATA_ROOT", str(data_root))
    return data_root


def test_manifest_is_deterministic_and_has_requested_sizes(benchmark_data: Path) -> None:
    left = build_subset_manifest()
    right = build_subset_manifest()
    assert left == right
    assert len(left["datasets"]["gsm8k"]["quality_indices"]) == 20
    assert len(left["datasets"]["humaneval"]["quality_indices"]) == 10
    assert len(left["datasets"]["gsm8k"]["parity_indices"]) == 5
    assert len(left["datasets"]["humaneval"]["parity_indices"]) == 5


def test_large_v2_manifest_is_disjoint_and_heads_are_nested(benchmark_data: Path) -> None:
    manifest = build_large_scale_manifest(data_root=benchmark_data)
    expected = {
        "gsm8k": ((800, 160, 200), (500, 100, 200), 159),
        "humaneval": ((96, 24, 32), (64, 16, 32), 12),
    }
    assert manifest["purpose"] == "large_v2_internal_treepc_training"
    assert manifest["split_policy"] == "head_splits_nested_in_matching_pc_splits"
    for name, (pc_sizes, head_sizes, unused) in expected.items():
        row = manifest["datasets"][name]
        pc = [set(row[f"pc_{split}_indices"]) for split in ("train", "validation", "test")]
        heads = [set(row[f"head_{split}_indices"]) for split in ("train", "validation", "test")]
        assert tuple(map(len, pc)) == pc_sizes
        assert tuple(map(len, heads)) == head_sizes
        assert not (pc[0] & pc[1] or pc[0] & pc[2] or pc[1] & pc[2])
        assert all(head.issubset(pc_split) for head, pc_split in zip(heads, pc))
        selected = row["pc_train_indices"] + row["pc_validation_indices"] + row["pc_test_indices"]
        assert len(selected) + unused == row["source_size"]


def test_large_v2_profile_matches_the_documented_budget() -> None:
    project_root = Path(__file__).parents[2]
    config = yaml.safe_load((project_root / "configs/large_v2/pipeline.yaml").read_text(encoding="utf-8"))
    assert config["profile"] == "large_v2"
    assert config["shared_pc_and_head_splits"] is False
    assert config["teacher"]["steps"] == 256
    assert config["pc_lora"]["decoding_steps"] == [8, 16, 32, 64]
    assert config["heads"]["decoding_steps"] == [8, 16, 32, 64]
    assert config["heads"]["candidate_size"] == {
        "gsm8k": {8: 32, 16: 16, 32: 8, 64: 4},
        "humaneval": {8: 64, 16: 32, 32: 16, 64: 8},
    }
    assert config["rq1"] == {
        "teacher_steps": 256,
        "student_steps": [8, 16, 32, 64],
        "state_source": "pc_lora_on_policy_mid_state",
        "recall_k": 16,
        "ece_bins": 15,
    }
