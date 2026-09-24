#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from treepc.data.pc_cache import (
    NestedPCDataset,
    build_nested_pc_bundle,
    validate_pc_bundle,
)
from treepc.dream.loader import load_dream
from treepc.models.pc_lora import create_pc_lora_student, load_pc_lora, lora_parameter_summary
from treepc.training.pc_trainer import evaluate_pc_student, train_pc_student
from treepc.utils.io import sha256_file, write_json
from treepc.utils.seed import seed_everything


def build_cache(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    bundle = build_nested_pc_bundle(args.trajectory)
    validate_pc_bundle(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    write_json(
        args.summary,
        {
            "schema": bundle["schema"],
            "dataset": bundle["dataset"],
            "partition": bundle["partition"],
            "record_count": len(bundle["records"]),
            "target_token_count": sum(
                int(record["anchor_positions"].numel()) for record in bundle["records"]
            ),
            "cache_path": str(output),
            "cache_sha256": sha256_file(output),
            "eligible_for_pc_training": bundle["eligible_for_pc_training"],
            "evaluation_only": bundle["evaluation_only"],
        },
    )


def train(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PC-LoRA training requires CUDA")
    seed_everything(int(config["seed"]))
    train_dataset = NestedPCDataset(
        args.train_caches, expected_partitions={"train", "pc_train"}
    )
    validation_dataset = NestedPCDataset(
        args.validation_caches, expected_partitions={"validation", "pc_validation"}
    )
    test_dataset = (
        NestedPCDataset(args.test_caches, expected_partitions={"test", "pc_test"})
        if args.test_caches
        else None
    )
    loaded = load_dream(device)
    student = create_pc_lora_student(
        loaded.model,
        rank=int(config["rank"]),
        alpha=int(config["alpha"]),
        dropout=float(config["dropout"]),
        target_modules=tuple(config["target_modules"]),
    )
    parameter_summary = lora_parameter_summary(student)
    baseline_validation = evaluate_pc_student(student, validation_dataset, device)
    baseline_test = evaluate_pc_student(student, test_dataset, device) if test_dataset else None
    report = train_pc_student(
        student,
        train_dataset,
        validation_dataset,
        device=device,
        output_dir=args.adapter_dir,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        gradient_accumulation=int(config["gradient_accumulation"]),
        seed=int(config["seed"]),
    )
    del student
    loaded.model = None
    torch.cuda.empty_cache()
    best_test = None
    if test_dataset:
        best_loaded = load_dream(device)
        best_student = load_pc_lora(best_loaded.model, args.adapter_dir)
        best_test = evaluate_pc_student(best_student, test_dataset, device)
        del best_student
        best_loaded.model = None
        torch.cuda.empty_cache()
    report.update(
        {
            "schema": "treepc.pc_lora_training.v1",
            "config": config,
            "parameter_summary": parameter_summary,
            "baseline_validation": baseline_validation,
            "baseline_test": baseline_test,
            "best_adapter_test": best_test,
            "train_state_count": len(train_dataset),
            "validation_state_count": len(validation_dataset),
            "test_state_count": len(test_dataset) if test_dataset else 0,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        }
    )
    write_json(args.report, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested PC labels or train Dream PC-LoRA")
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("build-cache")
    cache.add_argument("--trajectory", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--summary", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--train-caches", nargs="+", required=True)
    fit.add_argument("--validation-caches", nargs="+", required=True)
    fit.add_argument("--test-caches", nargs="+")
    fit.add_argument("--config", default="configs/stage4_pc/pc_lora.yaml")
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--adapter-dir", required=True)
    fit.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.command == "build-cache":
        build_cache(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
