#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from treepc.data.datasets import BenchmarkDataset, validate_large_scale_manifest
from treepc.data.pc_cache import (
    NestedPCDataset,
    build_nested_pc_bundle,
    validate_pc_bundle,
)
from treepc.dream.adapter import DreamAdapter
from treepc.dream.loader import load_dream
from treepc.models.pc_lora import create_pc_lora_student, load_pc_lora, lora_parameter_summary
from treepc.training.onpolicy_pc import train_mixed_budget_pc
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
    if args.teacher_steps is not None:
        config["teacher_steps"] = args.teacher_steps
    if args.student_budgets is not None:
        config["student_budgets"] = args.student_budgets
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PC-LoRA training requires CUDA")
    seed_everything(int(config["seed"]))
    train_dataset = NestedPCDataset(args.train_caches, expected_partitions={"train", "pc_train"})
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
    if config.get("training_mode") == "mixed_budget_on_policy":
        if not args.manifest:
            raise ValueError("Mixed-budget PC training requires --manifest (train prompt IDs)")
        manifest = json.loads(Path(args.manifest).read_text())
        validate_large_scale_manifest(manifest)
        for paths, split in ((args.train_caches, "train"), (args.validation_caches, "validation")):
            seen = set()
            for path in paths:
                cache = torch.load(path, map_location="cpu", weights_only=False)
                if int(cache["teacher_steps"]) != int(config["teacher_steps"]):
                    raise ValueError(f"Teacher budget mismatch in warm-up cache: {path}")
                if cache.get("source_manifest_fingerprint") != manifest["fingerprint"]:
                    raise ValueError(f"Warm-up cache manifest mismatch: {path}")
                name = cache["dataset"]
                seen.update((name, int(r["dataset_index"])) for r in cache["records"])
            expected = {
                (name, i)
                for name, entry in manifest["datasets"].items()
                for i in entry[f"pc_{split}_indices"]
            }
            if seen != expected:
                raise ValueError(f"Warm-up {split} prompt IDs differ from the manifest")
        prompts = []
        for name, entry in manifest["datasets"].items():
            dataset = BenchmarkDataset(name)
            if sha256_file(dataset.path) != entry["source_sha256"]:
                raise ValueError(f"Dataset changed since manifest: {name}")
            from treepc.evaluation.tasks import MAX_NEW_TOKENS

            length = MAX_NEW_TOKENS[name]
            prompts.extend((dataset.prompt(i), length) for i in entry["pc_train_indices"])
        loaded.model = student
        report = train_mixed_budget_pc(
            DreamAdapter(loaded), train_dataset, validation_dataset, prompts, config, args.adapter_dir
        )
    else:
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
            "model": loaded.metadata,
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
    fit.add_argument("--manifest", help="Fixed split manifest for current-student rollouts")
    fit.add_argument("--teacher-steps", type=int, help="Override the teacher budget recorded in this profile")
    fit.add_argument(
        "--student-budgets", nargs="+", type=int, help="Mixed-budget on-policy training schedules"
    )
    args = parser.parse_args()
    if args.command == "build-cache":
        build_cache(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
