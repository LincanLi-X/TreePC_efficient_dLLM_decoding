#!/usr/bin/env python
from __future__ import annotations

import argparse

from treepc.utils.environment import environment_report
from treepc.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--min-free-memory-gib", type=float, default=20.0)
    args = parser.parse_args()
    report = environment_report()
    if not report["cuda_available"] or len(report["gpus"]) < args.min_gpus:
        raise RuntimeError(f"Expected at least {args.min_gpus} visible CUDA GPU(s)")
    usable = [gpu for gpu in report["gpus"] if gpu["free_gib"] >= args.min_free_memory_gib]
    if len(usable) < args.min_gpus:
        raise RuntimeError(
            f"Only {len(usable)} GPU(s) have at least {args.min_free_memory_gib:.1f} GiB free"
        )
    write_json(args.output, report)
    print(f"environment recorded: {args.output}")


if __name__ == "__main__":
    main()
