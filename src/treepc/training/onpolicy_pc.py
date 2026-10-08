"""Two-phase PC-LoRA. Rollout buffers are CPU memory, never repository assets."""

import random
from contextlib import nullcontext

import torch

from treepc.posterior.consistency import posterior_consistency_kl
from treepc.posterior.topk_support import topk_log_probs_with_tail
from treepc.training.pc_trainer import evaluate_pc_student, pc_student_logits


def enable_checkpointing(student):
    base = student.get_base_model()
    if getattr(base.config, "model_type", "") == "llada":
        import importlib

        module = importlib.import_module(type(base.model).__module__)
        base.model.set_activation_checkpointing(module.ActivationCheckpointingStrategy.whole_layer)
    else:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()


class RolloutBuffer:
    def __init__(self, adapter, prompts, config, seed):
        self.adapter, self.prompts, self.config = adapter, prompts, config
        self.rng = random.Random(seed)
        self.records = []
        self.refreshes = 0

    @torch.no_grad()
    def refresh(self):
        adapter = self.adapter
        adapter.model.eval()
        records = []
        budgets = list(self.config["student_budgets"])
        if not budgets or any(not isinstance(b, int) or b < 1 for b in budgets):
            raise ValueError("student_budgets must contain positive integers")
        size = max(len(budgets), int(self.config.get("rollout_buffer_size", 16)))
        for ordinal in range(size):
            prompt, length = self.rng.choice(self.prompts)
            budget = budgets[ordinal] if ordinal < len(budgets) else self.rng.choice(budgets)
            state = adapter.make_initial_state(adapter.encode_prompt(prompt), length, budget)
            capture = self.rng.randrange(budget)
            ts = torch.linspace(1, 1e-3, budget + 1, device=adapter.device)
            for step in range(capture + 1):
                state.step_index, state.timestep_t, state.timestep_s = step, ts[step], ts[step + 1]
                out = adapter.forward_state(state)
                positions = out.masked_positions[0].nonzero().flatten()
                if step == capture and positions.numel():
                    max_anchors = int(self.config.get("max_anchors", 64))
                    if positions.numel() > max_anchors:
                        positions = positions[
                            torch.randperm(positions.numel(), device=positions.device)[:max_anchors]
                        ]
                    # Disabling LoRA recovers the frozen pretrained teacher without a second 7B copy.
                    with adapter.model.disable_adapter():
                        teacher = adapter.forward_state(state).aligned_logits[0, positions]
                    ids, probs, tail = topk_log_probs_with_tail(teacher, int(self.config.get("top_k", 16)))
                    records.append(
                        {
                            "input_ids": state.input_ids[0].detach().cpu().clone(),
                            "anchor_positions": positions.cpu(),
                            "teacher_topk_ids": ids.cpu(),
                            "teacher_topk_log_probs": probs.cpu(),
                            "teacher_tail_mass": tail.cpu(),
                            "budget": budget,
                            "step": step,
                        }
                    )
                    break
                confidence, tokens = adapter.propose_tokens_and_confidence(out, "entropy", 0.0, None, None)
                n = adapter.compute_commit_budget(
                    out.masked_positions, ts[step], ts[step + 1], step == budget - 1
                )
                selected = confidence.topk(n).indices
                state.input_ids[0, positions[selected]] = tokens[selected]
        if not records:
            raise RuntimeError("Rollout refresh produced no masked supervision states")
        self.records = records
        self.refreshes += 1

    def sample(self):
        budget = self.rng.choice(self.config["student_budgets"])
        candidates = [record for record in self.records if record["budget"] == budget]
        if not candidates:
            raise RuntimeError(f"No rollout state for sampled student budget {budget}")
        return self.rng.choice(candidates)


def train_mixed_budget_pc(adapter, warmup_dataset, validation_dataset, prompts, config, output_dir):
    if not prompts or not len(warmup_dataset) or not len(validation_dataset):
        raise ValueError("Nonempty train prompts, warm-up cache and validation cache are required")
    total = int(config["total_updates"])
    fraction = float(config.get("warmup_fraction", 0.2))
    refresh_interval = int(config.get("rollout_refresh_interval", 20))
    accumulation = int(config.get("gradient_accumulation", 4))
    if total < 2 or not 0 < fraction < 1 or min(refresh_interval, accumulation) < 1:
        raise ValueError("Require >=2 updates, 0<warmup_fraction<1 and positive intervals")
    warmup = max(1, min(total - 1, round(total * fraction)))
    student = adapter.model
    enable_checkpointing(student)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total)
    rng = random.Random(int(config["seed"]))
    buffer = RolloutBuffer(adapter, prompts, config, int(config["seed"]))
    history, best = [], float("inf")
    validation_interval = int(config.get("validation_interval", 20))
    for update in range(total):
        on_policy = update >= warmup
        if on_policy and (update - warmup) % refresh_interval == 0:
            buffer.refresh()
        student.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(accumulation):
            batch = buffer.sample() if on_policy else warmup_dataset[rng.randrange(len(warmup_dataset))]
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if adapter.device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = pc_student_logits(
                    student,
                    batch["input_ids"].unsqueeze(0).to(adapter.device),
                    batch["anchor_positions"].to(adapter.device),
                )
                loss, _ = posterior_consistency_kl(
                    logits,
                    batch["teacher_topk_ids"].to(adapter.device),
                    batch["teacher_topk_log_probs"].to(adapter.device),
                    batch["teacher_tail_mass"].to(adapter.device),
                )
            (loss / accumulation).backward()
            loss_value += float(loss.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        row = {"update": update + 1, "phase": "on_policy" if on_policy else "warmup", "loss": loss_value}
        print(row, flush=True)
        if (update + 1) % validation_interval == 0 or update + 1 == total:
            metrics = evaluate_pc_student(student, validation_dataset, adapter.device)
            row["validation"] = metrics
            if metrics["pc_kl"] < best:
                best = metrics["pc_kl"]
                student.save_pretrained(output_dir, safe_serialization=True)
        history.append(row)
    return {
        "history": history,
        "best_validation_kl": best,
        "rollout_refreshes": buffer.refreshes,
        "warmup_updates": warmup,
        "onpolicy_updates": total - warmup,
        "teacher_policy": "frozen_base_same_state",
        "buffer_storage": "memory_only",
    }
