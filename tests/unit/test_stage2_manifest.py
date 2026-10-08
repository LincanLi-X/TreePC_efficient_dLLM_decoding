import json
from pathlib import Path

import pytest

from treepc.data.datasets import build_oracle_manifest, build_stage3_manifest, extend_oracle_manifest


@pytest.fixture
def manifest_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    for name in ("gsm8k", "humaneval"):
        dataset_dir = data_root / name
        dataset_dir.mkdir(parents=True)
        rows = []
        for index in range(40):
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

    stage1 = {
        "schema_version": 1,
        "fingerprint": "stage1-fixture",
        "datasets": {name: {"quality_indices": [0, 1, 2, 3, 4]} for name in ("gsm8k", "humaneval")},
    }
    stage1_path = tmp_path / "stage1.json"
    stage1_path.write_text(json.dumps(stage1), encoding="utf-8")
    existing = build_oracle_manifest(stage1_path, seed=2026, gsm8k_size=2, humaneval_size=2)
    existing_path = tmp_path / "stage2.json"
    existing_path.write_text(json.dumps(existing), encoding="utf-8")
    return stage1_path, existing_path


def test_oracle_manifest_excludes_stage1_and_is_not_training_eligible(
    manifest_inputs: tuple[Path, Path],
) -> None:
    stage1_path, _ = manifest_inputs
    manifest = build_oracle_manifest(stage1_path, seed=2026, gsm8k_size=2, humaneval_size=2)
    assert manifest["eligible_for_head_training"] is False
    assert manifest["source_split"] == "test"
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    for name in ("gsm8k", "humaneval"):
        assert len(manifest["datasets"][name]["indices"]) == 2
        assert set(manifest["datasets"][name]["indices"]).isdisjoint(
            stage1["datasets"][name]["quality_indices"]
        )


def test_extended_oracle_manifest_preserves_existing_subset(
    manifest_inputs: tuple[Path, Path],
) -> None:
    stage1_path, existing_path = manifest_inputs
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    expanded = extend_oracle_manifest(existing_path, stage1_path, seed=2027)
    for name in ("gsm8k", "humaneval"):
        assert expanded["datasets"][name]["existing_indices"] == existing["datasets"][name]["indices"]
        assert len(expanded["datasets"][name]["indices"]) == 10
        assert len(set(expanded["datasets"][name]["indices"])) == 10


def test_stage3_manifest_is_disjoint_and_internal_only(
    manifest_inputs: tuple[Path, Path],
) -> None:
    stage1_path, stage2_path = manifest_inputs
    manifest = build_stage3_manifest(
        stage1_path,
        stage2_path,
        gsm8k_train_size=4,
        gsm8k_validation_size=2,
        humaneval_train_size=4,
        humaneval_validation_size=2,
    )
    assert manifest["eligible_for_head_training"] is True
    assert manifest["standard_benchmark_claim_allowed"] is False
    for value in manifest["datasets"].values():
        selected = set(value["train_indices"]) | set(value["validation_indices"])
        excluded = set(value["excluded_stage1_indices"]) | set(value["excluded_stage2_indices"])
        assert selected.isdisjoint(excluded)
        assert set(value["train_indices"]).isdisjoint(value["validation_indices"])
