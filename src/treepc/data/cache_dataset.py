from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor
from torch.utils.data import Dataset

from treepc.data.cache_schema import validate_counterfactual_bundle, validate_trajectory_bundle


def load_training_records(
    trajectory_paths: list[str | Path], counterfactual_paths: list[str | Path]
) -> list[dict[str, Any]]:
    hidden_by_key: dict[tuple[str, int], Tensor] = {}
    for path in trajectory_paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        validate_trajectory_bundle(bundle)
        if bundle.get("eligible_for_head_training") is not True:
            raise ValueError(f"Trajectory cache is not training eligible: {path}")
        for record in bundle["records"]:
            hidden_by_key[(record["sample_id"], int(record["step_index"]))] = record["aligned_hidden"]
    records: list[dict[str, Any]] = []
    for path in counterfactual_paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        validate_counterfactual_bundle(bundle)
        if bundle.get("eligible_for_head_training") is not True:
            raise ValueError(f"Counterfactual cache is not training eligible: {path}")
        for record in bundle["records"]:
            key = (record["sample_id"], int(record["step_index"]))
            if key not in hidden_by_key:
                raise ValueError(f"Missing aligned hidden state for {key}")
            if "support_ids" not in record:
                raise ValueError("Stage-3 correction labels require exact Top-K union support")
            merged = dict(record)
            merged["aligned_hidden"] = hidden_by_key[key]
            records.append(merged)
    return records


def load_labeled_records(
    paths: list[str | Path], *, expected_partition: str | None = None
) -> list[dict[str, Any]]:
    """Load PC-on-policy Teacher labels that already embed the Student hidden states."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        validate_counterfactual_bundle(bundle)
        if bundle.get("eligible_for_head_training") is not True:
            raise ValueError(f"Labeled cache is not training eligible: {path}")
        if expected_partition and bundle.get("partition") != expected_partition:
            raise ValueError(
                f"Expected partition {expected_partition!r}, got {bundle.get('partition')!r}: {path}"
            )
        for record in bundle.get("records", []):
            required = {
                "aligned_hidden",
                "support_ids",
                "directed_dependency",
                "symmetric_dependency",
            }
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Labeled record is missing {sorted(missing)}")
            key = (record["sample_id"], int(record["steps"]), int(record["step_index"]))
            if key in seen:
                raise ValueError(f"Duplicate labeled state {key}")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError("No labeled records were loaded")
    return records


class DependencyCacheDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        dataset_key = {"gsm8k": 1, "humaneval": 2}.get(str(record.get("dataset")), 0)
        record_key = (
            dataset_key * 1_000_000_000_000
            + int(record.get("dataset_index", index)) * 1_000_000
            + int(record.get("steps", 0)) * 1_000
            + int(record.get("step_index", 0))
        )
        return {
            "hidden": record["aligned_hidden"].float(),
            "positions": record["candidate_positions"].long(),
            "timestep": torch.tensor(record["timestep"], dtype=torch.float32),
            "target": record["symmetric_dependency"].float(),
            "record_key": torch.tensor(record_key, dtype=torch.long),
        }


@dataclass(frozen=True)
class CorrectionIndex:
    record: int
    parent: int
    sample: int
    child: int
    identity: bool


class CorrectionCacheDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        identity_fraction: float = 0.25,
        selected_pairs: dict[int, set[tuple[int, int]]] | None = None,
    ) -> None:
        self.records = records
        self.indices: list[CorrectionIndex] = []
        for record_index, record in enumerate(records):
            nodes = int(record["candidate_positions"].numel())
            samples = (
                int(record["sampled_parent_tokens"].shape[1])
                if record["sampled_parent_tokens"].ndim == 2
                else 1
            )
            all_pairs = [
                (parent, sample, child)
                for parent in range(nodes)
                for sample in range(samples)
                for child in range(nodes)
                if parent != child
            ]
            correction_pairs = all_pairs
            if selected_pairs is not None:
                selected = selected_pairs.get(record_index, set())
                correction_pairs = [pair for pair in all_pairs if (pair[0], pair[2]) in selected]
            self.indices.extend(
                CorrectionIndex(record_index, parent, sample, child, False)
                for parent, sample, child in correction_pairs
            )
            identity_count = int(round(len(correction_pairs) * identity_fraction))
            identity_candidates = all_pairs
            if selected_pairs is not None:
                identity_candidates = [pair for pair in all_pairs if (pair[0], pair[2]) not in selected]
            weakest = sorted(
                identity_candidates,
                key=lambda pair: self._directed_sample_value(record, *pair),
            )[:identity_count]
            self.indices.extend(
                CorrectionIndex(record_index, parent, sample, child, True)
                for parent, sample, child in weakest
            )

    @staticmethod
    def _directed_sample_value(record: dict[str, Any], parent: int, sample: int, child: int) -> float:
        if "directed_dependency_samples" in record:
            return float(record["directed_dependency_samples"][parent, sample, child])
        return float(record["directed_dependency"][parent, child])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.indices[index]
        record = self.records[item.record]
        parent, sample, child = item.parent, item.sample, item.child
        multi_sample = record["sampled_parent_tokens"].ndim == 2
        cache_index = (parent, sample, child) if multi_sample else (parent, child)
        base_support = record["base_support_log_probs"][cache_index].float()
        base_other = record["base_other_log_probs"][cache_index].float()
        if item.identity:
            target_support = base_support
            target_other = base_other
        else:
            target_support = record["conditional_support_log_probs"][cache_index].float()
            target_other = record["conditional_other_log_probs"][cache_index].float()
        positions = record["candidate_positions"]
        parent_token = (
            record["sampled_parent_tokens"][parent, sample]
            if multi_sample
            else record["sampled_parent_tokens"][parent]
        )
        return {
            "child_hidden": record["aligned_hidden"][child].float(),
            "parent_hidden": record["aligned_hidden"][parent].float(),
            "parent_token": parent_token.long(),
            "relative_position": (positions[child] - positions[parent]).float(),
            "sequence_length": torch.tensor(record["state_token_ids"].numel(), dtype=torch.float32),
            "timestep": torch.tensor(record["timestep"], dtype=torch.float32),
            "dependency_weight": record["symmetric_dependency"][parent, child].float(),
            "token_ids": record["support_ids"][cache_index].long(),
            "support_mask": record["support_mask"][cache_index].bool(),
            "base_support_log_probs": base_support,
            "base_other_log_prob": base_other,
            "target_support_log_probs": target_support,
            "target_other_log_prob": target_other,
            "identity": torch.tensor(item.identity),
        }


def load_dream_embedding_weights(model_dir: str | Path, device: str | torch.device) -> tuple[Tensor, Tensor]:
    model_dir = Path(model_dir)
    import json

    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))

    def load_key(key: str) -> Tensor:
        shard = model_dir / index["weight_map"][key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        return tensor.to(device=device, dtype=torch.bfloat16)

    return load_key("model.embed_tokens.weight"), load_key("lm_head.weight")
