from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from treepc.utils.io import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class LoadedDream:
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    metadata: dict[str, Any]


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs/model/dream_7b_instruct.yaml"
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {config_path}")
    return value


def resolve_model_dir(config: dict[str, Any] | None = None) -> Path:
    config = config or load_model_config()
    path = Path(os.environ.get("TREEPC_MODEL_DIR", config["local_dir"])).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str | torch.device = "auto", minimum_free_gib: float = 20.0) -> torch.device:
    if str(requested) != "auto":
        device = torch.device(requested)
        if device.type != "cuda":
            raise RuntimeError("Stage-1 Dream experiments require CUDA")
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    candidates: list[tuple[int, int]] = []
    for index in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(index)
        if free / 2**30 >= minimum_free_gib:
            candidates.append((free, index))
    if not candidates:
        raise RuntimeError(f"No GPU has at least {minimum_free_gib:.1f} GiB free")
    return torch.device(f"cuda:{max(candidates)[1]}")


def load_dream(
    device: str | torch.device = "auto",
    config_path: str | Path | None = None,
    model_revision: str | None = None,
) -> LoadedDream:
    config = load_model_config(config_path)
    model_dir = resolve_model_dir(config)
    if not (model_dir / "config.json").is_file() or not list(model_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"Incomplete Dream checkpoint: {model_dir}")
    resolved_device = resolve_device(device, float(config.get("minimum_free_memory_gib", 20)))
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        revision=model_revision,
        padding_side=str(config.get("padding_side", "left")),
    )
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        revision=model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attention_types = {layer.self_attn.__class__.__name__ for layer in model.model.layers}
    if attention_types != {"DreamSdpaAttention"}:
        raise RuntimeError(f"Dream SDPA fast path is not active: {sorted(attention_types)}")
    code_files = [
        model_dir / name
        for name in ("configuration_dream.py", "modeling_dream.py", "generation_utils.py")
    ]
    code_hash = hashlib.sha256("".join(sha256_file(path) for path in code_files).encode()).hexdigest()
    tokenizer_files = [model_dir / name for name in ("tokenizer_config.json", "vocab.json", "merges.txt")]
    tokenizer_hash = hashlib.sha256(
        "".join(sha256_file(path) for path in tokenizer_files).encode()
    ).hexdigest()
    metadata = {
        "repo_id": config["repo_id"],
        "local_dir": str(model_dir),
        "model_revision": model_revision,
        "remote_code_sha256": code_hash,
        "tokenizer_sha256": tokenizer_hash,
        "mask_token_id": int(model.config.mask_token_id),
        "pad_token_id": int(model.config.pad_token_id),
        "dtype": str(dtype),
        "device": str(resolved_device),
        "gpu_name": torch.cuda.get_device_name(resolved_device),
        "attention_classes": sorted(attention_types),
    }
    return LoadedDream(model, tokenizer, resolved_device, dtype, metadata)
