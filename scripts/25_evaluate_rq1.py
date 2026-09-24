#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from treepc.data.cache_schema import (
    validate_counterfactual_bundle,
    validate_trajectory_bundle,
)
from treepc.dream.loader import load_dream
from treepc.evaluation.rq1 import expected_calibration_error, posterior_metric_vectors
from treepc.models.pc_lora import load_pc_lora
from treepc.training.pc_trainer import pc_student_logits

MetricStore = dict[str, list[torch.Tensor]]


def _teacher_records(
    paths: list[str], expected_steps: int
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], str]:
    records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    fingerprint: str | None = None
    for path in paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        validate_trajectory_bundle(bundle)
        if int(bundle["teacher_steps"]) != expected_steps:
            raise ValueError(
                f"Expected a {expected_steps}-NFE Teacher cache, got "
                f"{bundle['teacher_steps']}: {path}"
            )
        current = bundle.get("manifest_fingerprint")
        if fingerprint is not None and current != fingerprint:
            raise ValueError("Teacher trajectory manifest fingerprints differ")
        fingerprint = current
        dataset = bundle["dataset"]
        for record in bundle["records"]:
            required = {
                "marginal_positions",
                "marginal_topk_ids",
                "marginal_topk_log_probs",
                "marginal_tail_mass",
            }
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"Teacher cache predates RQ1 marginal targets; missing {sorted(missing)}"
                )
            records[(dataset, record["sample_id"])].append(record)
    for values in records.values():
        values.sort(key=lambda row: float(row["timestep"]), reverse=True)
    if fingerprint is None:
        raise ValueError("No Teacher trajectory records were loaded")
    return records, fingerprint


def _new_store() -> MetricStore:
    return defaultdict(list)


def _append_metrics(store: MetricStore, values: dict[str, torch.Tensor]) -> None:
    for name, tensor in values.items():
        store[name].append(tensor.detach().float().cpu())


def _summarize(
    grouped: dict[tuple[str, int, str], MetricStore],
    metadata: dict[tuple[str, int, str], dict[str, int]],
    *,
    recall_k: int,
    ece_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        dataset, nfe, method = key
        values = {name: torch.cat(parts) for name, parts in grouped[key].items()}
        info = metadata[key]
        rows.append(
            {
                "dataset": dataset,
                "student_nfe": nfe,
                "method": method,
                "state_count": info["state_count"],
                "token_count": int(values["marginal_kl"].numel()),
                "comparable_anchor_rate": info["comparable_tokens"]
                / max(info["teacher_tokens"], 1),
                "teacher_to_student_marginal_kl": float(values["marginal_kl"].mean()),
                "teacher_top1_agreement": float(values["top1_agreement"].mean()),
                f"teacher_token_recall_at_{recall_k}": float(
                    values["teacher_token_recall"].mean()
                ),
                "student_entropy": float(values["student_entropy"].mean()),
                "student_top1_confidence": float(
                    values["student_top1_confidence"].mean()
                ),
                "student_teacher_top1_probability": float(
                    values["teacher_top1_probability"].mean()
                ),
                "student_teacher_top1_ece": expected_calibration_error(
                    values["student_top1_confidence"],
                    values["top1_agreement"],
                    bins=ece_bins,
                ),
            }
        )
    return rows


def _write_outputs(report: Path, csv_path: Path, payload: dict[str, Any]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = report.with_name(report.name + ".tmp")
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    report_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = payload["rows"]
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(report_tmp, report)
    os.replace(csv_tmp, csv_path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RQ1 marginal calibration on shared held-out PC states"
    )
    parser.add_argument("--student-state-caches", nargs="+", required=True)
    parser.add_argument("--teacher-trajectories", nargs="+", required=True)
    parser.add_argument("--pc-lora", required=True)
    parser.add_argument("--partition", choices=("head_validation", "head_test"), required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--teacher-steps", type=int, default=256)
    parser.add_argument("--recall-k", type=int, default=16)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    allowed_steps = set(args.steps)
    teacher_by_sample, teacher_fingerprint = _teacher_records(
        args.teacher_trajectories, args.teacher_steps
    )
    loaded = load_dream(args.device)
    student = load_pc_lora(loaded.model, args.pc_lora)
    student.eval()
    grouped: dict[tuple[str, int, str], MetricStore] = defaultdict(_new_store)
    metadata: dict[tuple[str, int, str], dict[str, int]] = defaultdict(
        lambda: {"state_count": 0, "teacher_tokens": 0, "comparable_tokens": 0}
    )
    skipped_states = 0
    try:
        for path in args.student_state_caches:
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            validate_counterfactual_bundle(bundle)
            if bundle.get("partition") != args.partition:
                raise ValueError(
                    f"Expected {args.partition!r}, got {bundle.get('partition')!r}: {path}"
                )
            if bundle.get("manifest_fingerprint") != teacher_fingerprint:
                raise ValueError("Student and Teacher caches use different manifests")
            dataset = bundle["dataset"]
            for record in bundle["records"]:
                nfe = int(record["steps"])
                if nfe not in allowed_steps:
                    continue
                candidates = teacher_by_sample.get((dataset, record["sample_id"]))
                if not candidates:
                    raise ValueError(f"Missing Teacher states for {dataset}/{record['sample_id']}")
                target = min(
                    candidates,
                    key=lambda row: abs(float(row["timestep"]) - float(record["timestep"])),
                )
                teacher_positions = target["marginal_positions"].long()
                state_ids = record["state_token_ids"].long()
                comparable = state_ids[teacher_positions].eq(int(loaded.model.config.mask_token_id))
                if not comparable.any():
                    skipped_states += 1
                    continue
                anchors = teacher_positions[comparable].to(loaded.device)
                input_ids = state_ids.unsqueeze(0).to(loaded.device)
                teacher_ids = target["marginal_topk_ids"][comparable].long().to(loaded.device)
                teacher_log_probs = (
                    target["marginal_topk_log_probs"][comparable].float().to(loaded.device)
                )
                teacher_tail = target["marginal_tail_mass"][comparable].float().to(loaded.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pc_logits = pc_student_logits(student, input_ids, anchors)
                    with student.disable_adapter():
                        dream_logits = pc_student_logits(student, input_ids, anchors)
                for method, logits in (("dream", dream_logits), ("pc_lora", pc_logits)):
                    key = (dataset, nfe, method)
                    _append_metrics(
                        grouped[key],
                        posterior_metric_vectors(
                            logits,
                            teacher_ids,
                            teacher_log_probs,
                            teacher_tail,
                            recall_k=args.recall_k,
                        ),
                    )
                    metadata[key]["state_count"] += 1
                    metadata[key]["teacher_tokens"] += int(teacher_positions.numel())
                    metadata[key]["comparable_tokens"] += int(comparable.sum())
            del bundle
            gc.collect()
    finally:
        del student
        loaded.model = None
        torch.cuda.empty_cache()

    rows = _summarize(
        grouped, metadata, recall_k=args.recall_k, ece_bins=args.ece_bins
    )
    if not rows:
        raise ValueError("RQ1 evaluation produced no rows")
    expected = {
        (dataset, nfe, method)
        for dataset in ("gsm8k", "humaneval")
        for nfe in allowed_steps
        for method in ("dream", "pc_lora")
    }
    actual = {(row["dataset"], row["student_nfe"], row["method"]) for row in rows}
    if actual != expected:
        raise ValueError(f"Incomplete RQ1 matrix: missing={sorted(expected - actual)}")
    payload = {
        "schema": "treepc.rq1_marginal_calibration.v1",
        "partition": args.partition,
        "teacher_nfe": args.teacher_steps,
        "student_nfe": sorted(allowed_steps),
        "recall_k": args.recall_k,
        "ece_bins": args.ece_bins,
        "state_source": "pc_lora_on_policy",
        "comparison_policy": "dream_and_pc_lora_on_the_same_state",
        "teacher_alignment": "nearest_normalized_progress_teacher_state",
        "anchor_policy": "teacher_masked_positions_still_masked_in_student_state",
        "manifest_fingerprint": teacher_fingerprint,
        "skipped_states_without_comparable_anchors": skipped_states,
        "rows": rows,
    }
    _write_outputs(Path(args.report), Path(args.csv), payload)


if __name__ == "__main__":
    main()
