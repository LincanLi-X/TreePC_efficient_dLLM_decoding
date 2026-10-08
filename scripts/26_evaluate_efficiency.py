#!/usr/bin/env python
"""One method per process: comparable timing/memory, measured rather than assumed NFE."""

import argparse
import json
import time
from pathlib import Path

import torch

from treepc.backbones.base import load_adapter
from treepc.data.datasets import BenchmarkDataset, validate_large_scale_manifest
from treepc.decoding.learned_treepc import learned_treepc_generate
from treepc.dream.generation import custom_independent_generate
from treepc.evaluation.external import ExternalDecoder
from treepc.evaluation.tasks import task_score
from treepc.models.pc_lora import load_pc_lora
from treepc.models.treepc_bundle import TreePCBundle
from treepc.utils.io import append_jsonl, sha256_file
from treepc.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=["vanilla", "pc_only", "local_treepc", "treepc", "fast_dllm", "cd4lm"],
    )
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval", "math500", "mbpp"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 8, 16, 32, 48, 64, 80, 96, 112])
    parser.add_argument("--teacher-steps", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--pc-lora")
    parser.add_argument("--dependency-checkpoint")
    parser.add_argument("--correction-checkpoint")
    parser.add_argument("--external-checkout")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5050)
    parser.add_argument("--profile-aux", action="store_true", help="Separate instrumented overhead run")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require warmup>=1, repeats>=1 and valid shard index/count")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads(Path(args.manifest).read_text())
    validate_large_scale_manifest(manifest)
    adapter = load_adapter(args.device)
    hook = None
    try:
        pc = args.method in {"pc_only", "local_treepc", "treepc"}
        if pc:
            if not args.pc_lora:
                parser.error("PC methods require --pc-lora")
            adapter.model = load_pc_lora(adapter.model, args.pc_lora)
        bundle = None
        if args.method in {"local_treepc", "treepc"}:
            if not args.dependency_checkpoint or not args.correction_checkpoint:
                parser.error("TreePC methods require both head checkpoints")
            bundle = TreePCBundle.from_checkpoints(
                adapter, args.dependency_checkpoint, args.correction_checkpoint
            )
        external = None
        if args.method in {"fast_dllm", "cd4lm"}:
            if not args.external_checkout:
                parser.error("External methods require --external-checkout")
            external = ExternalDecoder(adapter, args.method, args.external_checkout, args.threshold)
        count = [0]

        def record_forward(module, inputs):
            count[0] += 1

        hook = adapter.model.register_forward_pre_hook(record_forward)

        def generate(prompt, steps):
            if external:
                sequence = external(prompt, steps, args.max_new_tokens)
                encoded = adapter.encode_prompt(prompt)
                return sequence, adapter.decode(sequence, encoded["input_ids"].shape[1])[0], {}
            if bundle:
                result, _, timings = learned_treepc_generate(
                    adapter,
                    bundle,
                    prompt,
                    steps=steps,
                    max_new_tokens=args.max_new_tokens,
                    edge_source="local" if args.method == "local_treepc" else "learned",
                    profile_aux=args.profile_aux,
                )
            else:
                result = custom_independent_generate(
                    adapter, prompt, steps=steps, max_new_tokens=args.max_new_tokens
                )
                timings = {}
            return result.sequences, result.texts[0], timings

        for name in args.datasets:
            dataset = BenchmarkDataset(name)
            entry = manifest["datasets"][name]
            if sha256_file(dataset.path) != entry["source_sha256"]:
                raise ValueError(f"Dataset changed since manifest: {name}")
            indices = entry[f"pc_{args.split}_indices"][args.shard_index :: args.num_shards]
            for steps in args.steps:
                if not indices:
                    continue
                for _ in range(args.warmup):
                    generate(dataset.prompt(indices[0]), steps)
                for index in indices:
                    seed = args.seed + index + steps * 10_000
                    for repeat in range(args.repeats):
                        seed_everything(seed)
                        torch.cuda.synchronize(adapter.device)
                        torch.cuda.reset_peak_memory_stats(adapter.device)
                        before = count[0]
                        start = time.perf_counter()
                        sequence, text, timings = generate(dataset.prompt(index), steps)
                        torch.cuda.synchronize(adapter.device)
                        latency = time.perf_counter() - start
                        peak = torch.cuda.max_memory_allocated(adapter.device) / 2**20
                        nfe = count[0] - before
                        passed, error = task_score(name, dataset[index], text)
                        tokens = len(adapter.tokenizer.encode(text, add_special_tokens=False))
                        row = {
                            "backbone": adapter.metadata["backbone"],
                            "dataset": name,
                            "teacher_nfe": args.teacher_steps,
                            "steps": steps,
                            "method": args.method,
                            "sample_id": dataset[index]["sample_id"],
                            "repeat": repeat,
                            "seed": seed,
                            "split": args.split,
                            "actual_nfe": nfe,
                            "passed": passed,
                            "error": error,
                            "latency_s": latency,
                            "peak_gpu_memory_mib": peak,
                            "output_tokens": tokens,
                            "output_tokens_per_s": tokens / latency,
                            "max_new_tokens": args.max_new_tokens,
                            "batch_size": 1,
                            "gpu_name": torch.cuda.get_device_name(adapter.device),
                            "profile_aux": args.profile_aux,
                            "timing_source": "measured",
                            "manifest_fingerprint": manifest["fingerprint"],
                            "model": adapter.metadata,
                            "external_decoder": external.metadata if external else None,
                            "standard_benchmark_claim_allowed": manifest["standard_benchmark_claim_allowed"],
                            "timings": timings,
                            "output": text,
                        }
                        append_jsonl(destination, row)
                        del sequence
    finally:
        if hook is not None:
            hook.remove()
        adapter.close()


if __name__ == "__main__":
    main()
