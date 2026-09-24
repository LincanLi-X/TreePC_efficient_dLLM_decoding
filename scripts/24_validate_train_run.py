#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from treepc.data.datasets import validate_large_scale_manifest
from treepc.utils.io import sha256_file, write_json


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(require_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a completed from-scratch smoke run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = load_json(run_dir / "manifest.json")
    validate_large_scale_manifest(manifest)

    adapter_config = require_file(run_dir / "checkpoints/pc_lora/adapter_config.json")
    adapter_model = require_file(run_dir / "checkpoints/pc_lora/adapter_model.safetensors")
    dependency_path = require_file(run_dir / "checkpoints/dependency_head.pt")
    correction_path = require_file(run_dir / "checkpoints/correction_head.pt")
    dependency = torch.load(dependency_path, map_location="cpu", weights_only=False)
    correction = torch.load(correction_path, map_location="cpu", weights_only=False)
    if dependency.get("schema") != "treepc.dependency_head.v1":
        raise ValueError("The dependency checkpoint has an unsupported schema")
    if correction.get("schema") != "treepc.correction_head.v1":
        raise ValueError("The correction checkpoint has an unsupported schema")
    if "global_scale_logit" not in correction.get("state_dict", {}):
        raise ValueError("The new correction checkpoint is missing global_scale_logit")
    global_scale = float(
        torch.sigmoid(correction["state_dict"]["global_scale_logit"].float()).item()
    )
    if not 0.0 < global_scale < 1.0:
        raise ValueError(f"The learned global scale is out of range: {global_scale}")

    reports = {
        name: load_json(run_dir / f"reports/{name}.json")
        for name in ("pc_lora", "dependency_head", "correction_head", "heldout_heads")
    }
    require_file(run_dir / "reports/head_test/summary.json")
    require_file(run_dir / "reports/head_test/summary.csv")
    rows_path = require_file(run_dir / "eval/head_test/shard_0/rows.jsonl")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    expected_samples = sum(
        len(manifest["datasets"][dataset]["head_test_indices"])
        for dataset in ("gsm8k", "humaneval")
    )
    expected_methods = {
        "dream_baseline",
        "pc_only",
        "pc_local_treepc",
        "pc_learned_treepc",
    }
    expected_rows = expected_samples * len(set(args.steps)) * len(expected_methods)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} evaluation rows, found {len(rows)}")
    if {row["method"] for row in rows} != expected_methods:
        raise ValueError("Evaluation rows do not contain all four expected methods")
    if {int(row["steps"]) for row in rows} != set(args.steps):
        raise ValueError("Evaluation rows do not contain all requested NFE values")
    if any(int(row["dream_nfe"]) != int(row["steps"]) for row in rows):
        raise ValueError("At least one evaluation row has an inexact Dream NFE")

    write_json(
        args.output,
        {
            "schema": "treepc.train_smoke_complete.v1",
            "passed": True,
            "run_dir": str(run_dir),
            "manifest_fingerprint": manifest["fingerprint"],
            "evaluation_rows": len(rows),
            "evaluation_samples": expected_samples,
            "steps": sorted(set(args.steps)),
            "methods": sorted(expected_methods),
            "learned_global_scale": global_scale,
            "artifacts": {
                "pc_lora_adapter_config_sha256": sha256_file(adapter_config),
                "pc_lora_adapter_model_sha256": sha256_file(adapter_model),
                "dependency_head_sha256": sha256_file(dependency_path),
                "correction_head_sha256": sha256_file(correction_path),
            },
            "report_schemas": {name: report.get("schema") for name, report in reports.items()},
        },
    )
    print(f"TreePC from-scratch smoke run validated: {args.output}")


if __name__ == "__main__":
    main()
