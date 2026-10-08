"""Opt-in four-task driver regression with tiny toy records, not benchmark results."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from treepc.backbones.base import load_adapter
from treepc.data.datasets import BenchmarkDataset, build_large_scale_manifest
from treepc.data.pc_cache import build_nested_pc_bundle
from treepc.data.trajectory import collect_teacher_trajectory
from treepc.utils.io import write_json


@pytest.mark.model
@pytest.mark.cuda
def test_four_task_paper_driver(tmp_path, monkeypatch):
    root = os.environ.get("TREEPC_TEST_MODEL_ROOT")
    if not root:
        pytest.skip("Set TREEPC_TEST_MODEL_ROOT to opt into the real-model driver test")
    project = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    names = ["gsm8k", "humaneval", "math500", "mbpp"]
    for name in names:
        rows = []
        for i in range(3):
            rows.append(
                dict(
                    sample_id=f"{name}/{i}",
                    question=f"What is {i}+2?",
                    answer=str(i + 2),
                    prompt=f'def add_{i}(x):\n    """Add {i} to x."""\n',
                    test=f"def check(fn):\n    assert fn(1)=={i+1}",
                    entry_point=f"add_{i}",
                    problem=f"Evaluate {i}+2.",
                    text=f"Write add_{i}(x), adding {i} to x.",
                    test_list=[f"assert add_{i}(1)=={i+1}"],
                )
            )
        path = data / name / "samples.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setenv("TREEPC_BACKBONE", "dream")
    monkeypatch.setenv("TREEPC_MODEL_DIR", str(Path(root) / "DREAM-7B"))
    monkeypatch.setenv("TREEPC_DATA_ROOT", str(data))
    sizes = {name: {"train": 1, "validation": 1, "test": 1} for name in names}
    manifest = build_large_scale_manifest(pc_sizes=sizes, head_sizes=sizes, data_root=data)
    run_root = tmp_path / "runs"
    profile = run_root / "dream/teacher_4"
    write_json(profile / "manifest.json", manifest)
    adapter = load_adapter("cuda:0")
    try:
        for name in names:
            dataset = BenchmarkDataset(name, data)
            for split in ("train", "validation", "test"):
                index = manifest["datasets"][name][f"pc_{split}_indices"][0]
                records, _, _ = collect_teacher_trajectory(
                    adapter,
                    dataset.prompt(index),
                    dataset[index]["sample_id"],
                    index,
                    teacher_steps=4,
                    max_new_tokens=16,
                    states_per_example=4,
                    candidate_size=4,
                    top_k=16,
                )
                bundle = {
                    "schema": "treepc.teacher_trajectory.v1",
                    "dataset": name,
                    "partition": split,
                    "eligible_for_head_training": True,
                    "teacher_steps": 4,
                    "manifest_fingerprint": manifest["fingerprint"],
                    "records": records,
                }
                path = profile / "cache/teacher" / f"{name}_{split}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(bundle, path)
                if split != "test":
                    pc_path = profile / "cache/pc" / f"{name}_{split}.pt"
                    pc_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(build_nested_pc_bundle(path), pc_path)
    finally:
        adapter.close()
    config = yaml.safe_load((project / "configs/paper/experiments.yaml").read_text())
    config.update(
        backbones={"dream": {"model_dir": str(Path(root) / "DREAM-7B")}},
        teacher_steps=[4],
        student_steps=[2, 4],
        max_new_tokens=16,
        repeats=1,
        pc_splits=sizes,
        head_splits=sizes,
        external={},
    )
    for key in ("pc_config", "dependency_config", "correction_config"):
        original = yaml.safe_load((project / config[key]).read_text())
        if key == "pc_config":
            original.update(
                teacher_steps=4,
                total_updates=3,
                gradient_accumulation=1,
                student_budgets=[2, 4],
                rollout_buffer_size=2,
                rollout_refresh_interval=1,
                validation_interval=3,
                rank=2,
                alpha=4,
                max_anchors=4,
            )
        else:
            original.update(epochs=1, batch_size=2)
            if key == "correction_config":
                original["adaptation_epochs"] = 1
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(original))
        config[key] = str(path)
    config_path = tmp_path / "paper.yaml"
    config_path.write_text(yaml.safe_dump(config))
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/run_paper_experiments.py"),
            "--config",
            str(config_path),
            "--run-root",
            str(run_root),
            "--data-root",
            str(data),
            "--gpus",
            "0",
            "--prepare-heads",
        ],
        env=os.environ.copy(),
        cwd=project,
        check=True,
    )
    for report in (
        "rq1_test.csv",
        "rq2_test.csv",
        "heldout_heads.json",
        "rq3_test.csv",
        "rq4_matched_quality.json",
        "rq4_overhead.csv",
    ):
        assert (profile / "reports" / report).is_file()
