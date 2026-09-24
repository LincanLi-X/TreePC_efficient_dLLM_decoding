from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from treepc.data.cache_schema import validate_trajectory_bundle


def build_nested_pc_bundle(trajectory_path: str | Path) -> dict[str, Any]:
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    validate_trajectory_bundle(trajectory)
    if trajectory.get("eligible_for_head_training") is not True:
        raise ValueError("PC labels must come from a training-eligible trajectory bundle")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in trajectory["records"]:
        grouped[record["sample_id"]].append(record)
    records: list[dict[str, Any]] = []
    for sample_id, sample_records in grouped.items():
        sample_records.sort(key=lambda row: int(row["step_index"]))
        for coarse, fine in zip(sample_records[:-1], sample_records[1:]):
            anchor_positions = fine["candidate_positions"].long()
            fine_mask_values = fine["state_token_ids"][anchor_positions]
            if not coarse["state_token_ids"][anchor_positions].eq(fine_mask_values).all():
                raise ValueError(f"Nested PC anchors are not masked in coarse state for {sample_id}")
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset_index": int(coarse["dataset_index"]),
                    "student_state_token_ids": coarse["state_token_ids"].long(),
                    "generation_mask": coarse["generation_mask"].bool(),
                    "prompt_length": int(coarse["prompt_length"]),
                    "student_step_index": int(coarse["step_index"]),
                    "student_timestep": float(coarse["timestep"]),
                    "teacher_step_index": int(fine["step_index"]),
                    "teacher_timestep": float(fine["timestep"]),
                    "anchor_positions": anchor_positions,
                    "teacher_topk_ids": fine["base_topk_ids"].long(),
                    "teacher_topk_log_probs": fine["base_topk_log_probs"].float(),
                    "teacher_tail_mass": fine["base_tail_mass"].float(),
                    "teacher_final_tokens": fine["final_teacher_tokens"][anchor_positions].long(),
                }
            )
    partition = trajectory.get("partition")
    is_training = partition in {"train", "pc_train"}
    return {
        "schema": "treepc.pc_nested.v1",
        "dataset": trajectory["dataset"],
        "partition": partition,
        "eligible_for_pc_cache": True,
        "eligible_for_pc_training": is_training,
        "evaluation_only": not is_training,
        "source_manifest_fingerprint": trajectory.get("manifest_fingerprint"),
        "teacher_steps": trajectory["teacher_steps"],
        "records": records,
    }


def validate_pc_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema") != "treepc.pc_nested.v1":
        raise ValueError("Unsupported PC cache schema")
    if bundle.get("eligible_for_pc_cache", bundle.get("eligible_for_pc_training")) is not True:
        raise ValueError("PC cache is not eligible for PC supervision/evaluation")
    for record in bundle["records"]:
        anchors = int(record["anchor_positions"].numel())
        if record["teacher_topk_ids"].shape[0] != anchors:
            raise ValueError("PC target/anchor mismatch")
        if record["teacher_topk_log_probs"].shape != record["teacher_topk_ids"].shape:
            raise ValueError("PC target id/log-probability mismatch")


class NestedPCDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        paths: list[str | Path],
        *,
        expected_partitions: set[str] | None = None,
    ) -> None:
        self.records: list[dict[str, Any]] = []
        for path in paths:
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            validate_pc_bundle(bundle)
            if expected_partitions is not None and bundle.get("partition") not in expected_partitions:
                raise ValueError(
                    f"Expected PC cache partition in {sorted(expected_partitions)}, "
                    f"got {bundle.get('partition')!r}: {path}"
                )
            self.records.extend(bundle["records"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        return {
            "input_ids": record["student_state_token_ids"],
            "anchor_positions": record["anchor_positions"],
            "teacher_topk_ids": record["teacher_topk_ids"],
            "teacher_topk_log_probs": record["teacher_topk_log_probs"],
            "teacher_tail_mass": record["teacher_tail_mass"],
            "teacher_final_tokens": record["teacher_final_tokens"],
        }
