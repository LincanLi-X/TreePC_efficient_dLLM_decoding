from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model


def create_pc_lora_student(
    model: torch.nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> PeftModel:
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        inference_mode=False,
    )
    student = get_peft_model(model, config)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_("lora_" in name)
    return student


def load_pc_lora(model: torch.nn.Module, adapter_dir: str | Path) -> PeftModel:
    student = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    student.to(dtype=torch.bfloat16)
    student.eval()
    return student


def lora_parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }
