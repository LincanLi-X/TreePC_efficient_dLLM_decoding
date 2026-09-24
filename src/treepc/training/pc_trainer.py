from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from treepc.dream.alignment import align_dream_hidden
from treepc.posterior.consistency import posterior_consistency_kl


def pc_student_logits(student: PeftModel, input_ids: Tensor, anchor_positions: Tensor) -> Tensor:
    dream = student.get_base_model()
    output = dream.model(
        input_ids=input_ids,
        attention_mask="full",
        position_ids=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    aligned_hidden = align_dream_hidden(output.last_hidden_state)
    selected_hidden = aligned_hidden[0, anchor_positions]
    return dream.lm_head(selected_hidden)


@torch.inference_mode()
def evaluate_pc_student(
    student: PeftModel,
    dataset: Dataset[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    kls: list[Tensor] = []
    agreements = 0
    targets = 0
    final_nll: list[Tensor] = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        input_ids = batch["input_ids"].to(device)
        anchors = batch["anchor_positions"][0].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = pc_student_logits(student, input_ids, anchors)
        _, per_token = posterior_consistency_kl(
            logits,
            batch["teacher_topk_ids"][0].to(device),
            batch["teacher_topk_log_probs"][0].to(device),
            batch["teacher_tail_mass"][0].to(device),
        )
        kls.append(per_token.cpu())
        teacher_top1 = batch["teacher_topk_ids"][0, :, 0].to(device)
        agreements += int(logits.argmax(-1).eq(teacher_top1).sum().item())
        targets += int(anchors.numel())
        final_targets = batch["teacher_final_tokens"][0].to(device)
        final_nll.append(-torch.log_softmax(logits.float(), -1).gather(
            -1, final_targets.unsqueeze(-1)
        ).squeeze(-1).cpu())
    values = torch.cat(kls)
    return {
        "pc_kl": float(values.mean().item()),
        "pc_kl_median": float(values.median().item()),
        "teacher_top1_agreement": agreements / max(targets, 1),
        "teacher_final_token_nll": float(torch.cat(final_nll).mean().item()),
        "target_token_count": targets,
    }


def train_pc_student(
    student: PeftModel,
    train_dataset: Dataset[dict[str, Tensor]],
    validation_dataset: Dataset[dict[str, Tensor]],
    *,
    device: torch.device,
    output_dir: str | Path,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    gradient_accumulation: int = 4,
    seed: int = 4040,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    total_updates = max(1, math.ceil(len(loader) * epochs / gradient_accumulation))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    student.enable_input_require_grads()
    best_kl = math.inf
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        student.train()
        running = 0.0
        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)
            anchors = batch["anchor_positions"][0].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = pc_student_logits(student, input_ids, anchors)
                loss, _ = posterior_consistency_kl(
                    logits,
                    batch["teacher_topk_ids"][0].to(device),
                    batch["teacher_topk_log_probs"][0].to(device),
                    batch["teacher_tail_mass"][0].to(device),
                )
                scaled_loss = loss / gradient_accumulation
            scaled_loss.backward()
            running += float(loss.detach().item())
            if step % gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 12 == 0 or step == len(loader):
                print(
                    f"pc epoch={epoch}/{epochs} state={step}/{len(loader)} "
                    f"loss={running / step:.5f}",
                    flush=True,
                )
        metrics = evaluate_pc_student(student, validation_dataset, device)
        history.append({"epoch": float(epoch), "train_loss": running / len(loader), **metrics})
        print(
            f"pc validation epoch={epoch} kl={metrics['pc_kl']:.5f} "
            f"top1={metrics['teacher_top1_agreement']:.4f}",
            flush=True,
        )
        if metrics["pc_kl"] < best_kl:
            best_kl = metrics["pc_kl"]
            best_metrics = metrics
            student.save_pretrained(output_dir, safe_serialization=True)
    return {"best_validation": best_metrics, "history": history, "adapter_dir": str(output_dir)}
