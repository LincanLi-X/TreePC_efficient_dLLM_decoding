#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from treepc.data.cache_dataset import (
    CorrectionCacheDataset,
    DependencyCacheDataset,
    load_dream_embedding_weights,
    load_labeled_records,
)
from treepc.dream.loader import load_model_config, resolve_model_dir
from treepc.models.correction_head import ConditionalCorrectionHead
from treepc.models.dependency_head import DependencyHead
from treepc.training.correction_trainer import evaluate_correction_head
from treepc.training.dependency_trainer import evaluate_dependency_head
from treepc.utils.io import write_json

TREE_BASELINES = ("learned", "local", "hidden_cosine", "random", "oracle")


def build_rq2_row(
    dataset: str,
    student_nfe: int,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one dataset/NFE dependency evaluation for JSON and CSV."""
    candidate_sizes = [int(record["candidate_positions"].numel()) for record in records]
    row: dict[str, Any] = {
        "dataset": dataset,
        "student_nfe": student_nfe,
        "state_count": len(records),
        "candidate_size_mean": sum(candidate_sizes) / len(candidate_sizes),
        "mae": metrics["mae"],
        "huber": metrics["huber"],
        "spearman": metrics["spearman"],
        "kendall_tau_b": metrics["kendall_tau_b"],
        "raw_symmetry_mae": metrics["raw_symmetry_mae"],
    }
    for tree_name in TREE_BASELINES:
        tree_metrics = metrics["trees"][tree_name]
        row[f"{tree_name}_mst_edge_overlap"] = tree_metrics["mst_edge_overlap"]
        row[f"{tree_name}_oracle_weight_capture"] = tree_metrics["oracle_weight_capture"]
    return row


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained heads on held-out PC states")
    parser.add_argument("--test-labeled", nargs="+", required=True)
    parser.add_argument("--dependency-checkpoint", required=True)
    parser.add_argument("--correction-checkpoint", required=True)
    parser.add_argument("--correction-config", default="configs/large_v2/correction_head.yaml")
    parser.add_argument("--steps", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", required=True)
    parser.add_argument("--rq2-report")
    parser.add_argument("--rq2-csv")
    args = parser.parse_args()
    if bool(args.rq2_report) != bool(args.rq2_csv):
        raise ValueError("Provide both --rq2-report and --rq2-csv, or neither")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Head evaluation requires CUDA")

    records = load_labeled_records(args.test_labeled, expected_partition="head_test")
    dependency_value = torch.load(args.dependency_checkpoint, map_location="cpu", weights_only=False)
    dependency = DependencyHead(**dependency_value["config"])
    dependency.load_compatible_state_dict(dependency_value["state_dict"])
    dependency.to(device)

    correction_value = torch.load(args.correction_checkpoint, map_location="cpu", weights_only=False)
    correction_config = yaml.safe_load(Path(args.correction_config).read_text(encoding="utf-8"))
    model_config = load_model_config()
    input_embedding, output_embedding = load_dream_embedding_weights(resolve_model_dir(model_config), device)
    correction = ConditionalCorrectionHead(input_embedding, output_embedding, **correction_value["config"])
    correction.load_compatible_state_dict(correction_value["state_dict"])
    correction.to(device)

    by_dataset = {
        name: [record for record in records if record.get("dataset") == name]
        for name in ("gsm8k", "humaneval")
    }
    metrics = {}
    for name, selected in (*by_dataset.items(), ("combined", records)):
        metrics[name] = {
            "state_count": len(selected),
            "dependency": evaluate_dependency_head(
                dependency,
                DataLoader(DependencyCacheDataset(selected), batch_size=1, shuffle=False),
                device,
            ),
            "correction": evaluate_correction_head(
                correction,
                DataLoader(
                    CorrectionCacheDataset(
                        selected,
                        identity_fraction=float(correction_config["identity_fraction"]),
                    ),
                    batch_size=int(correction_config["batch_size"]),
                    shuffle=False,
                ),
                device,
            ),
        }

    rq2_rows = []
    for dataset in ("gsm8k", "humaneval"):
        for student_nfe in args.steps:
            selected = [
                record
                for record in records
                if record.get("dataset") == dataset and int(record["steps"]) == student_nfe
            ]
            if not selected:
                raise ValueError(f"RQ2 has no held-out states for dataset={dataset}, NFE={student_nfe}")
            dependency_metrics = evaluate_dependency_head(
                dependency,
                DataLoader(DependencyCacheDataset(selected), batch_size=1, shuffle=False),
                device,
            )
            rq2_rows.append(build_rq2_row(dataset, student_nfe, selected, dependency_metrics))
    write_json(
        args.report,
        {
            "schema": "treepc.heldout_head_evaluation.v1",
            "partition": "head_test",
            "standard_benchmark_claim_allowed": False,
            "metrics": metrics,
        },
    )
    if args.rq2_report:
        write_json(
            args.rq2_report,
            {
                "schema": "treepc.rq2_dependency_recovery.v1",
                "partition": "head_test",
                "grouping": ["dataset", "student_nfe"],
                "student_nfe": args.steps,
                "tree_baselines": list(TREE_BASELINES),
                "rank_metrics": ["spearman", "kendall_tau_b"],
                "oracle_target": "teacher_counterfactual_dependency",
                "standard_benchmark_claim_allowed": False,
                "rows": rq2_rows,
            },
        )
        write_csv(args.rq2_csv, rq2_rows)


if __name__ == "__main__":
    main()
