#!/usr/bin/env python
"""Paper matrix orchestration. Intermediate inputs are external, validated artifacts."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import yaml

from treepc.data.datasets import build_large_scale_manifest
from treepc.utils.io import sha256_file, write_json

PROJECT = Path(__file__).resolve().parents[1]
STAGES = ["pc", "heads", "rq1", "rq2", "eval", "efficiency", "external", "reports"]


def build_jobs(config, stage, root, profile, teacher, gpus, split):
    cache, checkpoint, reports = root / "cache", root / "checkpoints", root / "reports"
    manifest = root / "manifest.json"
    pc = checkpoint / "pc_lora"
    dep, corr = checkpoint / "dependency_head.pt", checkpoint / "correction_head.pt"
    steps = list(map(str, config["student_steps"]))
    datasets = config["datasets"]
    jobs = []

    def add(script, arguments, required=(), gpu=None):
        jobs.append(
            {
                "command": [sys.executable, str(PROJECT / "scripts" / script), *map(str, arguments)],
                "required": [str(p) for p in required],
                "gpu": gpu if gpu is not None else gpus[0],
            }
        )

    def labeled(partition):
        return [cache / "heads" / f"{name}_{partition}_{i}.pt" for name in datasets for i in range(len(gpus))]

    if stage == "pc":
        if (pc / "adapter_model.safetensors").exists():
            return []  # Never overwrite a supplied trained PC adapter implicitly.
        train = [cache / "pc" / f"{name}_train.pt" for name in datasets]
        valid = [cache / "pc" / f"{name}_validation.pt" for name in datasets]
        add(
            "09_train_pc_lora.py",
            [
                "train",
                "--train-caches",
                *train,
                "--validation-caches",
                *valid,
                "--manifest",
                manifest,
                "--config",
                config["pc_config"],
                "--teacher-steps",
                teacher,
                "--student-budgets",
                *steps,
                "--adapter-dir",
                pc,
                "--report",
                reports / "pc.json",
            ],
            [manifest, *train, *valid],
        )
    elif stage == "collect-heads":
        for partition in ("train", "validation", "test"):
            for name in datasets:
                for i, gpu in enumerate(gpus):
                    state = cache / "heads" / f"{name}_{partition}_{i}.states.pt"
                    label = cache / "heads" / f"{name}_{partition}_{i}.pt"
                    add(
                        "10_recalibrate_heads.py",
                        [
                            "collect",
                            "--dataset",
                            name,
                            "--split",
                            partition,
                            "--manifest",
                            manifest,
                            "--pc-lora",
                            pc,
                            "--steps",
                            *steps,
                            "--shard-index",
                            i,
                            "--num-shards",
                            len(gpus),
                            "--output",
                            state,
                        ],
                        [manifest, pc / "adapter_model.safetensors"],
                        gpu,
                    )
                    add(
                        "10_recalibrate_heads.py",
                        ["label", "--input", state, "--output", label, "--top-k", 16, "--parent-samples", 3],
                        [state],
                        gpu,
                    )
    elif stage == "heads":
        for script, destination, conf in [
            ("06_train_dependency_head.py", dep, "dependency_config"),
            ("07_train_correction_head.py", corr, "correction_config"),
        ]:
            add(
                script,
                [
                    "--train-labeled",
                    *labeled("train"),
                    "--validation-labeled",
                    *labeled("validation"),
                    "--config",
                    config[conf],
                    "--checkpoint",
                    destination,
                    "--report",
                    reports / f"{destination.stem}.json",
                    *(["--dependency-checkpoint", dep] if destination == corr else []),
                ],
                [*labeled("train"), *labeled("validation"), *([dep] if destination == corr else [])],
            )
    elif stage == "rq1":
        teacher_paths = [cache / "teacher" / f"{name}_test.pt" for name in datasets]
        add(
            "25_evaluate_rq1.py",
            [
                "--teacher-trajectories",
                *teacher_paths,
                "--student-state-caches",
                *labeled("test"),
                "--pc-lora",
                pc,
                "--partition",
                "head_test",
                "--steps",
                *steps,
                "--teacher-steps",
                teacher,
                "--report",
                reports / "rq1_test.json",
                "--csv",
                reports / "rq1_test.csv",
            ],
            [*teacher_paths, *labeled("test"), pc / "adapter_model.safetensors"],
        )
    elif stage == "rq2":
        add(
            "22_evaluate_heads.py",
            [
                "--test-labeled",
                *labeled("test"),
                "--steps",
                *steps,
                "--dependency-checkpoint",
                dep,
                "--correction-checkpoint",
                corr,
                "--report",
                reports / "heldout_heads.json",
                "--rq2-report",
                reports / "rq2_test.json",
                "--rq2-csv",
                reports / "rq2_test.csv",
            ],
            [*labeled("test"), dep, corr],
        )
    elif stage in {"eval", "efficiency", "external"}:
        methods = (
            ["vanilla", "pc_only", "local_treepc", "treepc"]
            if stage != "external"
            else list(config["external"])
        )
        for method in methods:
            if stage == "external" and profile not in config["external"][method]["backbones"]:
                print(f"Unsupported official baseline: {method}/{profile}; not reported as implemented")
                continue
            budgets = list(dict.fromkeys(steps + ([str(teacher)] if method == "vanilla" else [])))
            for i, gpu in enumerate(gpus):
                output = root / "eval" / f"{stage}_{split}_{method}_{i}.jsonl"
                if output.exists():
                    raise FileExistsError(
                        f"Use a new run root or explicitly archive the existing output: {output}"
                    )
                args = [
                    "--method",
                    method,
                    "--datasets",
                    *datasets,
                    "--steps",
                    *budgets,
                    "--manifest",
                    manifest,
                    "--output",
                    output,
                    "--split",
                    split,
                    "--teacher-steps",
                    teacher,
                    "--max-new-tokens",
                    config["max_new_tokens"],
                    "--warmup",
                    config.get("warmup", 1),
                    "--repeats",
                    config.get("repeats", 3),
                    "--shard-index",
                    i,
                    "--num-shards",
                    len(gpus),
                ]
                required = [manifest]
                if method in {"pc_only", "local_treepc", "treepc"}:
                    args += ["--pc-lora", pc]
                    required += [pc / "adapter_model.safetensors"]
                if method in {"local_treepc", "treepc"}:
                    args += ["--dependency-checkpoint", dep, "--correction-checkpoint", corr]
                    required += [dep, corr]
                if stage == "efficiency":
                    args += ["--profile-aux"]
                if stage == "external":
                    args += [
                        "--external-checkout",
                        config["external"][method]["checkout"],
                        "--threshold",
                        config["external"][method]["threshold"],
                    ]
                add("26_evaluate_efficiency.py", args, required, gpu)
                jobs[-1]["method"] = method
    elif stage == "reports":
        files = sorted((root / "eval").glob("*.jsonl"))
        add("27_build_paper_reports.py", ["--inputs", *files, "--output-dir", reports], files)
    return jobs


def check_head_provenance(paths, pc, backbone, manifest=None, budgets=None):
    digest = sha256_file(pc / "adapter_model.safetensors")
    seen = {}
    for path in paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if bundle.get("pc_adapter_sha256") != digest or bundle.get("backbone") != backbone:
            raise ValueError(f"Stale/unverified on-policy cache {path}; regenerate from the final PC adapter")
        if manifest is not None:
            if bundle.get("manifest_fingerprint") != manifest["fingerprint"]:
                raise ValueError(f"On-policy cache manifest mismatch: {path}")
            name, partition = bundle["dataset"], bundle["partition"]
            expected = set(manifest["datasets"][name][f"{partition}_indices"])
            observed = seen.setdefault((name, partition), set())
            for record in bundle["records"]:
                identity = (int(record["dataset_index"]), int(record["steps"]))
                if identity[0] not in expected or (budgets is not None and identity[1] not in budgets):
                    raise ValueError(f"Head state does not belong to the configured split/budget: {path}")
                if identity in observed:
                    raise ValueError(f"Duplicate head state {name}/{partition}/{identity}")
                observed.add(identity)
    if manifest is not None and budgets is not None:
        for (name, partition), observed in seen.items():
            expected = {
                (i, budget) for i in manifest["datasets"][name][f"{partition}_indices"] for budget in budgets
            }
            if observed != expected:
                raise ValueError(f"Incomplete on-policy cache for {name}/{partition}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/experiments.yaml")
    parser.add_argument("--run-root", required=True, help="External artifacts root, NOT inside TreePC_Code")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3", help="Logical CUDA IDs visible to this process")
    parser.add_argument("--stage", choices=["all", "manifest", "collect-heads", *STAGES], default="all")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    parser.add_argument(
        "--prepare-heads", action="store_true", help="Opt-in runtime generation outside repository"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.run_root).expanduser().resolve()
    if root.is_relative_to(PROJECT):
        raise ValueError("All runtime artifacts must be outside TreePC_Code")
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU lanes must be unique")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_map = {gpu: gpu for gpu in gpus}
    if visible is not None:
        exposed = visible.split(",")
        if any(not gpu.isdigit() or int(gpu) >= len(exposed) for gpu in gpus):
            raise ValueError("--gpus refers to logical indices within CUDA_VISIBLE_DEVICES")
        gpu_map = {gpu: exposed[int(gpu)] for gpu in gpus}
    stages = STAGES if args.stage == "all" else [args.stage]
    if args.prepare_heads and "heads" in stages:
        stages = stages[: stages.index("heads")] + ["collect-heads"] + stages[stages.index("heads") :]
    for backbone, settings in config["backbones"].items():
        for teacher in config["teacher_steps"]:
            profile = root / backbone / f"teacher_{teacher}"
            env = dict(
                os.environ,
                TREEPC_BACKBONE=backbone,
                TREEPC_MODEL_DIR=str(Path(settings["model_dir"]).resolve()),
                TREEPC_DATA_ROOT=str(Path(args.data_root).resolve()),
                TREEPC_MAX_NEW_TOKENS=str(config["max_new_tokens"]),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=str(PROJECT / "src"),
            )
            if not args.dry_run:
                manifest = build_large_scale_manifest(
                    seed=config["seed"],
                    pc_sizes=config["pc_splits"],
                    head_sizes=config["head_splits"],
                    data_root=args.data_root,
                    protocol=config["protocol"],
                )
                if not (profile / "manifest.json").exists():
                    write_json(profile / "manifest.json", manifest)
                elif json.loads((profile / "manifest.json").read_text()) != manifest:
                    raise ValueError("Existing manifest differs from config/data; use a new run root")
            for stage in stages:
                splits = (
                    ["validation", "test"]
                    if args.split == "both" and stage in {"eval", "external", "efficiency"}
                    else ["test" if args.split == "both" else args.split]
                )
                for split in splits:
                    jobs = build_jobs(config, stage, profile, backbone, teacher, gpus, split)

                    def execute(job):
                        print(json.dumps(job), flush=True)
                        if args.dry_run:
                            return
                        missing = [path for path in job["required"] if not Path(path).exists()]
                        if missing:
                            raise FileNotFoundError(f"Configure the documented external inputs: {missing}")
                        if stage in {"heads", "rq1", "rq2"}:
                            head_paths = [
                                p for p in job["required"] if "/cache/heads/" in p and p.endswith(".pt")
                            ]
                            check_head_provenance(
                                head_paths,
                                profile / "checkpoints/pc_lora",
                                backbone,
                                json.loads((profile / "manifest.json").read_text()),
                                config["student_steps"],
                            )
                        job_env = dict(env, CUDA_VISIBLE_DEVICES=gpu_map[job["gpu"]])
                        if job.get("method") == "cd4lm":
                            job_env["TREEPC_MODEL_DIR"] = str(
                                Path(config["external"]["cd4lm"]["model_dir"]).resolve()
                            )
                        log_dir = profile / "logs"
                        log_dir.mkdir(parents=True, exist_ok=True)
                        with (
                            log_dir / f"{stage}_{split}_{job.get('method', 'stage')}_{job['gpu']}.log"
                        ).open("a") as log:
                            subprocess.run(
                                job["command"],
                                env=job_env,
                                cwd=PROJECT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )

                    # One worker per GPU, sequential jobs on that GPU. Head/PC stages remain single-GPU.
                    if stage in {"eval", "external", "efficiency", "collect-heads"}:

                        def lane(gpu):
                            for job in jobs:
                                if job["gpu"] == gpu:
                                    execute(job)

                        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                            list(pool.map(lane, gpus))
                    else:
                        for job in jobs:
                            execute(job)


if __name__ == "__main__":
    main()
