#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from treepc.data.datasets import build_large_scale_manifest
from treepc.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the reproducible large-v2 remote split")
    parser.add_argument("--config", default="configs/large_v2/pipeline.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = build_large_scale_manifest(
        seed=int(config["seed"]),
        pc_sizes=config["pc_lora"]["splits"],
        head_sizes=config["heads"]["splits"],
        require_identical_splits=bool(config.get("shared_pc_and_head_splits", False)),
    )
    output = Path(args.output)
    if args.validate_existing:
        if not output.is_file():
            raise FileNotFoundError(output)
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                "Existing manifest does not match the current large-v2 split config; "
                "use a new TREEPC_RUN_DIR."
            )
        print(f"validated existing manifest: {output}")
    else:
        if output.exists():
            raise FileExistsError(output)
        write_json(output, manifest)
    counts = {
        dataset: {
            key: len(value)
            for key, value in row.items()
            if key.endswith("_indices")
        }
        for dataset, row in manifest["datasets"].items()
    }
    print(f"large-scale manifest: {args.output}")
    print(counts)


if __name__ == "__main__":
    main()
