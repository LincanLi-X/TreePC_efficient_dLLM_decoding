#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import torch

from treepc.data.datasets import BenchmarkDataset
from treepc.dream.loader import load_model_config, resolve_model_dir
from treepc.utils.io import sha256_file, write_json


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_model(model_dir: Path) -> dict[str, Any]:
    required = [
        "config.json",
        "model.safetensors.index.json",
        "configuration_dream.py",
        "modeling_dream.py",
        "generation_utils.py",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    for name in required:
        require_file(model_dir / name)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError(f"No model shards are listed in {index_path}")
    shard_paths = [require_file(model_dir / name) for name in shards]
    return {
        "path": str(model_dir),
        "hidden_size": int(config["hidden_size"]),
        "index_sha256": sha256_file(index_path),
        "shards": [
            {"name": path.name, "bytes": path.stat().st_size} for path in shard_paths
        ],
        "total_shard_bytes": sum(path.stat().st_size for path in shard_paths),
    }


def validate_checkpoints(checkpoint_dir: Path) -> dict[str, Any]:
    adapter_config = require_file(checkpoint_dir / "pc_lora" / "adapter_config.json")
    adapter_model = require_file(checkpoint_dir / "pc_lora" / "adapter_model.safetensors")
    dependency_path = require_file(checkpoint_dir / "dependency_head.pt")
    correction_path = require_file(checkpoint_dir / "correction_head.pt")
    dependency = torch.load(dependency_path, map_location="cpu", weights_only=False)
    correction = torch.load(correction_path, map_location="cpu", weights_only=False)
    if dependency.get("schema") != "treepc.dependency_head.v1":
        raise ValueError(f"Unsupported dependency checkpoint: {dependency_path}")
    if correction.get("schema") != "treepc.correction_head.v1":
        raise ValueError(f"Unsupported correction checkpoint: {correction_path}")
    scale_logit = correction["state_dict"].get("global_scale_logit")
    legacy_scale_fallback = scale_logit is None
    global_scale = (
        float(correction["config"].get("global_scale_init", 0.05))
        if legacy_scale_fallback
        else float(torch.sigmoid(scale_logit.float()).item())
    )
    if not 0.0 < global_scale < 1.0:
        raise ValueError(f"Invalid correction global scale: {global_scale}")
    return {
        "path": str(checkpoint_dir),
        "pc_lora": {
            "adapter_config_sha256": sha256_file(adapter_config),
            "adapter_model_sha256": sha256_file(adapter_model),
        },
        "dependency_sha256": sha256_file(dependency_path),
        "correction_sha256": sha256_file(correction_path),
        "dependency_hidden_size": dependency["config"]["hidden_size"],
        "correction_hidden_size": correction["config"]["hidden_size"],
        "global_scale": global_scale,
        "legacy_global_scale_fallback": legacy_scale_fallback,
        "dependency_threshold": correction.get("tree_gate_calibration", {}).get(
            "dependency_threshold"
        ),
    }


def validate_datasets(minimum_rows: int) -> dict[str, Any]:
    result = {}
    for name in ("gsm8k", "humaneval"):
        dataset = BenchmarkDataset(name)
        if len(dataset) < minimum_rows:
            raise ValueError(f"{name} has {len(dataset)} rows; expected at least {minimum_rows}")
        result[name] = {
            "path": str(dataset.path),
            "rows": len(dataset),
            "sha256": sha256_file(dataset.path),
        }
    return result


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "accelerate", "peft", "safetensors", "PyYAML")
    versions = {name: importlib.metadata.version(name) for name in names}
    expected = {
        "torch": "2.11.0",
        "transformers": "4.48.0",
        "accelerate": "1.9.0",
        "peft": "0.14.0",
        "safetensors": "0.5.2",
        "PyYAML": "6.0.3",
    }
    mismatches = {
        name: {"expected": value, "actual": versions[name]}
        for name, value in expected.items()
        if not versions[name].split("+")[0] == value
    }
    if mismatches:
        raise RuntimeError(f"Python package version mismatch: {mismatches}")
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate assets for a one-GPU TreePC run")
    parser.add_argument(
        "--checkpoint-dir",
        help="Pretrained TreePC checkpoint directory; omit for a from-scratch training run",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-dataset-rows", type=int, default=2)
    parser.add_argument("--minimum-free-memory-gib", type=float, default=20.0)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The smoke job expects exactly one visible CUDA GPU")
    free, total = torch.cuda.mem_get_info(0)
    free_gib = free / 2**30
    if free_gib < args.minimum_free_memory_gib:
        raise RuntimeError(
            f"Visible GPU has only {free_gib:.2f} GiB free; "
            f"expected at least {args.minimum_free_memory_gib:.2f} GiB"
        )

    model = validate_model(resolve_model_dir(load_model_config()))
    checkpoints = (
        validate_checkpoints(Path(args.checkpoint_dir).expanduser().resolve())
        if args.checkpoint_dir
        else None
    )
    hidden_sizes = {int(model["hidden_size"])}
    if checkpoints:
        hidden_sizes.update(
            {
                int(checkpoints["dependency_hidden_size"]),
                int(checkpoints["correction_hidden_size"]),
            }
        )
    if hidden_sizes != {3584}:
        raise ValueError(f"Dream/checkpoint hidden-size mismatch: {sorted(hidden_sizes)}")

    report = {
        "schema": (
            "treepc.smoke_preflight.v1"
            if checkpoints
            else "treepc.train_smoke_preflight.v1"
        ),
        "mode": "pretrained_inference" if checkpoints else "from_scratch_training",
        "passed": True,
        "python_packages": package_versions(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "free_gib": free_gib,
            "total_gib": total / 2**30,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "model": model,
        "datasets": validate_datasets(args.minimum_dataset_rows),
    }
    if checkpoints:
        report["checkpoints"] = checkpoints
    if not report["gpu"]["bf16_supported"]:
        raise RuntimeError("The visible GPU does not support bfloat16")
    write_json(args.output, report)
    print(f"TreePC smoke preflight passed: {args.output}")


if __name__ == "__main__":
    main()
