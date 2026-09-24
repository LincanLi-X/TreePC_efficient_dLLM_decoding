from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/processed/track1_general"
DATASET_FILES = {
    "gsm8k": Path("gsm8k/samples.jsonl"),
    "humaneval": Path("humaneval/samples.jsonl"),
}


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve benchmark data from an argument, environment variable, or repository default."""
    path = Path(data_root or os.environ.get("TREEPC_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# Backward-compatible aliases for existing scripts/tests. New code should use
# BenchmarkDataset so TREEPC_DATA_ROOT is resolved explicitly at construction.
DATA_ROOT = resolve_data_root()
DATASETS = {name: DATA_ROOT / relative for name, relative in DATASET_FILES.items()}


class BenchmarkDataset:
    def __init__(self, name: str, data_root: str | Path | None = None) -> None:
        if name not in DATASET_FILES:
            raise ValueError(f"Unsupported dataset {name!r}")
        self.name = name
        self.path = resolve_data_root(data_root) / DATASET_FILES[name]
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Missing {name} data: {self.path}. Set TREEPC_DATA_ROOT on the remote host."
            )
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def prompt(self, index: int) -> str:
        row = self.rows[index]
        if self.name == "gsm8k":
            return (
                "Solve this grade-school math problem. Show concise reasoning and end with exactly one line "
                "in the form 'Final answer: <number>'.\n\n"
                f"Problem:\n{row['question']}\n\nSolution:\n"
            )
        return (
            "Complete the following Python function. Return only valid Python code for the function body or "
            f"continuation, without markdown.\n\n{row['prompt']}"
        )


def build_subset_manifest(seed: int = 42, gsm8k_size: int = 20, humaneval_size: int = 10) -> dict[str, Any]:
    rng = random.Random(seed)
    result: dict[str, Any] = {"schema_version": 1, "seed": seed, "datasets": {}}
    for name, size, parity_size in (("gsm8k", gsm8k_size, 5), ("humaneval", humaneval_size, 5)):
        dataset = BenchmarkDataset(name)
        indices = sorted(rng.sample(range(len(dataset)), size))
        sample_ids = [dataset[index]["sample_id"] for index in indices]
        source_hash = hashlib.sha256(dataset.path.read_bytes()).hexdigest()
        result["datasets"][name] = {
            "source_path": str(dataset.path),
            "source_sha256": source_hash,
            "source_size": len(dataset),
            "quality_indices": indices,
            "quality_sample_ids": sample_ids,
            "parity_indices": indices[:parity_size],
            "parity_sample_ids": sample_ids[:parity_size],
        }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return result


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported subset manifest")
    return value


def build_oracle_manifest(
    stage1_manifest_path: str | Path,
    seed: int = 2026,
    gsm8k_size: int = 2,
    humaneval_size: int = 2,
) -> dict[str, Any]:
    stage1 = load_manifest(stage1_manifest_path)
    rng = random.Random(seed)
    result: dict[str, Any] = {
        "schema_version": 2,
        "seed": seed,
        "purpose": "stage2_oracle_diagnostic_only",
        "eligible_for_head_training": False,
        "source_split": "test",
        "stage1_manifest_fingerprint": stage1["fingerprint"],
        "datasets": {},
    }
    for name, size in (("gsm8k", gsm8k_size), ("humaneval", humaneval_size)):
        dataset = BenchmarkDataset(name)
        excluded = set(stage1["datasets"][name]["quality_indices"])
        available = [index for index in range(len(dataset)) if index not in excluded]
        indices = sorted(rng.sample(available, size))
        result["datasets"][name] = {
            "indices": indices,
            "sample_ids": [dataset[index]["sample_id"] for index in indices],
            "source_path": str(dataset.path),
            "source_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
        }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return result


def extend_oracle_manifest(
    existing_manifest_path: str | Path,
    stage1_manifest_path: str | Path,
    seed: int = 2027,
    gsm8k_total: int = 10,
    humaneval_total: int = 10,
) -> dict[str, Any]:
    existing = json.loads(Path(existing_manifest_path).read_text(encoding="utf-8"))
    stage1 = load_manifest(stage1_manifest_path)
    if existing.get("schema_version") != 2:
        raise ValueError("Expected an existing stage-2 diagnostic manifest")
    rng = random.Random(seed)
    result: dict[str, Any] = {
        "schema_version": 2,
        "seed": seed,
        "purpose": "stage2_oracle_diagnostic_only_expanded",
        "eligible_for_head_training": False,
        "source_split": "test",
        "stage1_manifest_fingerprint": stage1["fingerprint"],
        "extends_manifest_fingerprint": existing["fingerprint"],
        "datasets": {},
    }
    for name, total in (("gsm8k", gsm8k_total), ("humaneval", humaneval_total)):
        dataset = BenchmarkDataset(name)
        existing_indices = list(existing["datasets"][name]["indices"])
        if total < len(existing_indices):
            raise ValueError(f"Requested total for {name} is smaller than the existing subset")
        excluded = set(stage1["datasets"][name]["quality_indices"]) | set(existing_indices)
        available = [index for index in range(len(dataset)) if index not in excluded]
        added_indices = sorted(rng.sample(available, total - len(existing_indices)))
        indices = existing_indices + added_indices
        result["datasets"][name] = {
            "indices": indices,
            "sample_ids": [dataset[index]["sample_id"] for index in indices],
            "existing_indices": existing_indices,
            "added_indices": added_indices,
            "added_sample_ids": [dataset[index]["sample_id"] for index in added_indices],
            "source_path": str(dataset.path),
            "source_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
        }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return result


def build_stage3_manifest(
    stage1_manifest_path: str | Path,
    stage2_manifest_path: str | Path,
    seed: int = 3030,
    gsm8k_train_size: int = 24,
    gsm8k_validation_size: int = 8,
    humaneval_train_size: int = 24,
    humaneval_validation_size: int = 8,
) -> dict[str, Any]:
    """Create a disjoint internal-only split for the small Stage-3 proof of concept.

    Both available source files are benchmark test splits.  The manifest therefore
    labels the resulting split as pseudo-train/pseudo-validation and explicitly
    forbids standard benchmark claims.  Stage-1 evaluation IDs and all Stage-2
    diagnostic IDs are excluded before sampling.
    """
    stage1 = load_manifest(stage1_manifest_path)
    stage2 = json.loads(Path(stage2_manifest_path).read_text(encoding="utf-8"))
    if stage2.get("schema_version") != 2:
        raise ValueError("Expected a Stage-2 manifest")
    rng = random.Random(seed)
    sizes = {
        "gsm8k": (gsm8k_train_size, gsm8k_validation_size),
        "humaneval": (humaneval_train_size, humaneval_validation_size),
    }
    result: dict[str, Any] = {
        "schema_version": 3,
        "seed": seed,
        "purpose": "stage3_internal_poc_head_training",
        "source_split": "benchmark_test_repurposed_internal_only",
        "eligible_for_head_training": True,
        "standard_benchmark_claim_allowed": False,
        "stage1_manifest_fingerprint": stage1["fingerprint"],
        "stage2_manifest_fingerprint": stage2["fingerprint"],
        "datasets": {},
    }
    for name, (train_size, validation_size) in sizes.items():
        dataset = BenchmarkDataset(name)
        stage1_excluded = set(stage1["datasets"][name]["quality_indices"])
        stage2_excluded = set(stage2["datasets"][name]["indices"])
        excluded = stage1_excluded | stage2_excluded
        available = [index for index in range(len(dataset)) if index not in excluded]
        requested = train_size + validation_size
        if requested > len(available):
            raise ValueError(f"Requested {requested} {name} samples but only {len(available)} remain")
        selected = rng.sample(available, requested)
        train_indices = sorted(selected[:train_size])
        validation_indices = sorted(selected[train_size:])
        result["datasets"][name] = {
            "source_path": str(dataset.path),
            "source_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
            "source_size": len(dataset),
            "train_indices": train_indices,
            "train_sample_ids": [dataset[index]["sample_id"] for index in train_indices],
            "validation_indices": validation_indices,
            "validation_sample_ids": [dataset[index]["sample_id"] for index in validation_indices],
            "excluded_stage1_indices": sorted(stage1_excluded),
            "excluded_stage2_indices": sorted(stage2_excluded),
        }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return result


LARGE_SCALE_PC_SIZES = {
    "gsm8k": {"train": 800, "validation": 160, "test": 200},
    "humaneval": {"train": 96, "validation": 24, "test": 32},
}
LARGE_SCALE_HEAD_SIZES = {
    "gsm8k": {"train": 500, "validation": 100, "test": 200},
    "humaneval": {"train": 64, "validation": 16, "test": 32},
}


def build_large_scale_manifest(
    seed: int = 5050,
    pc_sizes: dict[str, dict[str, int]] | None = None,
    head_sizes: dict[str, dict[str, int]] | None = None,
    data_root: str | Path | None = None,
    require_identical_splits: bool = False,
) -> dict[str, Any]:
    """Build the reproducible large-v2 split used by the remote run.

    Head splits are nested in the matching PC-LoRA splits while train,
    validation, and test remain mutually disjoint.
    """
    pc_sizes = pc_sizes or LARGE_SCALE_PC_SIZES
    head_sizes = head_sizes or LARGE_SCALE_HEAD_SIZES
    rng = random.Random(seed)
    result: dict[str, Any] = {
        "schema_version": 4,
        "seed": seed,
        "purpose": "large_v2_internal_treepc_training",
        "source_split": "benchmark_test_repurposed_internal_only",
        "standard_benchmark_claim_allowed": False,
        "split_policy": (
            "shared_pc_and_head_splits"
            if require_identical_splits
            else "head_splits_nested_in_matching_pc_splits"
        ),
        "datasets": {},
    }
    for name in ("gsm8k", "humaneval"):
        dataset = BenchmarkDataset(name, data_root=data_root)
        pc = pc_sizes[name]
        heads = head_sizes[name]
        if require_identical_splits and heads != pc:
            raise ValueError(f"{name}: shared PC/Head splits require identical requested sizes")
        for split in ("train", "validation", "test"):
            if heads[split] > pc[split]:
                raise ValueError(f"{name} head {split} size exceeds its PC-LoRA split")
        requested = sum(pc.values())
        if requested > len(dataset):
            raise ValueError(f"Requested {requested} {name} rows but source has {len(dataset)}")

        selected = rng.sample(range(len(dataset)), requested)
        boundaries = (pc["train"], pc["train"] + pc["validation"])
        pc_indices = {
            "train": sorted(selected[: boundaries[0]]),
            "validation": sorted(selected[boundaries[0] : boundaries[1]]),
            "test": sorted(selected[boundaries[1] :]),
        }
        if require_identical_splits:
            head_indices = {split: list(indices) for split, indices in pc_indices.items()}
        else:
            head_indices = {
                split: sorted(rng.sample(pc_indices[split], heads[split]))
                for split in ("train", "validation", "test")
            }
        entry: dict[str, Any] = {
            "source_path": str(dataset.path),
            "source_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
            "source_size": len(dataset),
        }
        for family, indices_by_split in (("pc", pc_indices), ("head", head_indices)):
            for split, indices in indices_by_split.items():
                entry[f"{family}_{split}_indices"] = indices
                entry[f"{family}_{split}_sample_ids"] = [
                    dataset[index]["sample_id"] for index in indices
                ]
        result["datasets"][name] = entry

    validate_large_scale_manifest(result)
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return result


def validate_large_scale_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 4:
        raise ValueError("Expected large-scale manifest schema_version=4")
    for name, entry in manifest.get("datasets", {}).items():
        pc = {split: set(entry[f"pc_{split}_indices"]) for split in ("train", "validation", "test")}
        heads = {
            split: set(entry[f"head_{split}_indices"])
            for split in ("train", "validation", "test")
        }
        if pc["train"] & pc["validation"] or pc["train"] & pc["test"] or pc["validation"] & pc["test"]:
            raise ValueError(f"{name}: PC-LoRA splits overlap")
        for split in ("train", "validation", "test"):
            if not heads[split].issubset(pc[split]):
                raise ValueError(f"{name}: head {split} is not nested in PC-LoRA {split}")
            if manifest.get("split_policy") == "shared_pc_and_head_splits" and heads[split] != pc[split]:
                raise ValueError(f"{name}: PC-LoRA and Head {split} splits must be identical")
        if (
            heads["train"] & heads["validation"]
            or heads["train"] & heads["test"]
            or heads["validation"] & heads["test"]
        ):
            raise ValueError(f"{name}: head splits overlap")
