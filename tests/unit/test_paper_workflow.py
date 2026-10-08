import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from treepc.backbones.llada import LLaDAAdapter
from treepc.data.datasets import BenchmarkDataset, build_large_scale_manifest
from treepc.evaluation.external import import_file
from treepc.evaluation.tasks import task_score

PROJECT = Path(__file__).resolve().parents[2]


def test_native_llada_transfer_quota():
    loaded = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(mask_token_id=99)),
        tokenizer=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        metadata={"backbone": "llada"},
    )
    adapter = LLaDAAdapter(loaded)
    state = adapter.make_initial_state({"input_ids": torch.tensor([[1, 2]])}, 17, 4)
    counts = [adapter.compute_commit_budget(state.generation_mask, None, None, i == 3) for i in range(4)]
    assert counts == [5, 4, 4, 4]


def test_four_dataset_standard_manifest(tmp_path):
    names = ("gsm8k", "humaneval", "math500", "mbpp")
    for name in names:
        rows = [
            dict(
                sample_id=f"{name}/{i}",
                question=f"question {i}",
                prompt=f"def f{i}():",
                problem=f"problem {i}",
                text=f"task {i}",
                test_list=[f"assert f{i}()==0"],
                source_split="train" if i < 4 else "test",
            )
            for i in range(6)
        ]
        path = tmp_path / name / "samples.jsonl"
        path.parent.mkdir()
        path.write_text("\n".join(json.dumps(row) for row in rows))
        assert BenchmarkDataset(name, tmp_path).prompt(0)
    sizes = {name: {"train": 2, "validation": 1, "test": 2} for name in names}
    manifest = build_large_scale_manifest(
        pc_sizes=sizes, head_sizes=sizes, data_root=tmp_path, protocol="standard"
    )
    assert manifest["standard_benchmark_claim_allowed"]
    for entry in manifest["datasets"].values():
        assert set(entry["pc_train_indices"]).issubset(range(4))
        assert set(entry["pc_validation_indices"]).issubset(range(4))
        assert set(entry["pc_test_indices"]) == {4, 5}


def test_paper_plan_covers_numeric_matrix_and_all_tasks(tmp_path):
    module = import_file(PROJECT / "scripts/run_paper_experiments.py", "test_paper_driver")
    config = yaml.safe_load((PROJECT / "configs/paper/experiments.yaml").read_text())
    for backbone in config["backbones"]:
        for teacher in (256, 512):
            jobs = module.build_jobs(config, "eval", tmp_path, backbone, teacher, ["0", "1"], "test")
            assert len(jobs) == 8
            for job in jobs:
                command = job["command"]
                assert all(task in command for task in ("gsm8k", "humaneval", "math500", "mbpp"))
                assert all(str(nfe) in command for nfe in config["student_steps"])


def test_quality_budget_selection_uses_validation_not_test():
    module = import_file(PROJECT / "scripts/27_build_paper_reports.py", "test_paper_reports")
    base = dict(
        backbone="dream",
        dataset="gsm8k",
        teacher_nfe=256,
        gpu_name="test GPU",
        max_new_tokens=512,
        profile_aux=False,
        manifest_fingerprint="fixed",
        sample_ids=["a", "b"],
    )
    rows = []
    for split in ("validation", "test"):
        for method, steps, score, latency in [
            ("vanilla", 256, 0.8, 10),
            ("treepc", 8, 0.5, 1),
            ("treepc", 16, 0.8 if split == "validation" else 0.6, 2),
        ]:
            rows.append(dict(base, split=split, method=method, steps=steps, score=score, latency_s=latency))
    selected = module.matched_quality(rows, 0.01)
    assert selected[0]["steps"] == 16 and selected[0]["speedup"] == 5
    assert not selected[0]["test_quality_within_tolerance"]


def test_mbpp_grading_and_gsm8k():
    sample = {"test_list": ["assert double(3)==6", "assert double(0)==0"]}
    assert task_score("mbpp", sample, "def double(x):\n    return x*2")[0]
    assert not task_score("mbpp", sample, "def double(x):\n    return x")[0]
    assert task_score("gsm8k", {"answer": "6"}, "Final answer: 6")[0]


def test_math500_equivalence():
    pytest.importorskip("math_verify")
    assert task_score("math500", {"answer": r"\frac{1}{2}"}, r"\boxed{0.5}")[0]
    assert not task_score("math500", {"answer": "2"}, r"\boxed{3}")[0]


def test_head_cache_provenance_rejects_cross_split_state(tmp_path):
    from treepc.utils.io import sha256_file

    module = import_file(PROJECT / "scripts/run_paper_experiments.py", "test_paper_provenance")
    pc = tmp_path / "pc"
    pc.mkdir()
    (pc / "adapter_model.safetensors").write_bytes(b"test-identity-not-model")
    bundle = dict(
        pc_adapter_sha256=sha256_file(pc / "adapter_model.safetensors"),
        backbone="dream",
        manifest_fingerprint="fixed",
        dataset="gsm8k",
        partition="head_train",
        records=[dict(dataset_index=1, steps=4)],
    )
    path = tmp_path / "cache.pt"
    torch.save(bundle, path)
    manifest = {"fingerprint": "fixed", "datasets": {"gsm8k": {"head_train_indices": [1]}}}
    module.check_head_provenance([str(path)], pc, "dream", manifest, [4])
    bundle["records"][0]["dataset_index"] = 2
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="split/budget"):
        module.check_head_provenance([str(path)], pc, "dream", manifest, [4])
