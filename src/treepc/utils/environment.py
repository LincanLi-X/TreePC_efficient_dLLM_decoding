from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import transformers

from treepc.dream.loader import load_model_config
from treepc.utils.io import sha256_file


def environment_report() -> dict[str, Any]:
    config = load_model_config()
    model_dir = Path(os.environ.get("TREEPC_MODEL_DIR", config["local_dir"])).expanduser()
    gpus = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        gpus.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "free_gib": free / 2**30,
                "total_gib": total / 2**30,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    files = [
        model_dir / name
        for name in ("config.json", "model.safetensors.index.json", "generation_utils.py")
    ]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
        "model_dir": str(model_dir),
        "model_files": {path.name: sha256_file(path) for path in files},
        "stage": 1,
        "full_step_reference_enabled": False,
    }


def print_environment() -> None:
    print(json.dumps(environment_report(), indent=2))
