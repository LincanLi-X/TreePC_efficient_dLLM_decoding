from __future__ import annotations

import time
from typing import Any

import torch
from torch.nn import functional as F

from treepc.dream.adapter import DreamAdapter, logits_digest, sample_tokens
from treepc.types import GenerationResult, StepTrace, clone_cpu


@torch.inference_mode()
def official_generate(
    adapter: DreamAdapter,
    prompt: str,
    *,
    steps: int,
    max_new_tokens: int,
    alg: str = "entropy",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    alg_temp: float | None = 0.0,
    capture_trace: bool = False,
) -> GenerationResult:
    encoded = adapter.encode_prompt(prompt)
    traces: list[StepTrace] = []
    pending: dict[str, Any] = {}

    def logits_hook(step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if capture_trace:
            mask = x.eq(adapter.mask_token_id)
            _, proposals = sample_tokens(logits[mask], temperature, top_p, top_k, alg)
            pending.clear()
            pending.update(
                pre_state=clone_cpu(x),
                proposals=clone_cpu(proposals),
                digest=logits_digest(logits, mask),
            )
        return logits

    def tokens_hook(step: int | None, x: torch.Tensor, logits: torch.Tensor | None) -> torch.Tensor:
        if capture_trace and step is not None:
            pre = pending["pre_state"]
            post = clone_cpu(x)
            changed = pre.ne(post) & pre.eq(adapter.mask_token_id)
            positions = changed.nonzero(as_tuple=False)[:, 1]
            tokens = post[0, positions]
            traces.append(
                StepTrace(step, pre, post, positions, tokens, pending["proposals"], pending["digest"])
            )
        return x

    torch.cuda.synchronize(adapter.device)
    started = time.perf_counter()
    output = adapter.model.diffusion_generate(
        encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        steps=steps,
        eps=1e-3,
        alg=alg,
        alg_temp=alg_temp,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_dict_in_generate=True,
        output_history=capture_trace,
        generation_logits_hook_func=logits_hook,
        generation_tokens_hook_func=tokens_hook,
    )
    torch.cuda.synchronize(adapter.device)
    latency = time.perf_counter() - started
    prompt_length = encoded["input_ids"].shape[1]
    return GenerationResult(
        sequences=output.sequences,
        texts=adapter.decode(output.sequences, prompt_length),
        traces=traces,
        latency_seconds=latency,
        effective_nfe=steps,
    )


@torch.inference_mode()
def custom_independent_generate(
    adapter: DreamAdapter,
    prompt: str,
    *,
    steps: int,
    max_new_tokens: int,
    alg: str = "entropy",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    alg_temp: float | None = 0.0,
    capture_trace: bool = False,
) -> GenerationResult:
    encoded = adapter.encode_prompt(prompt)
    state = adapter.make_initial_state(encoded, max_new_tokens, steps)
    x = state.input_ids
    timesteps = torch.linspace(1, 1e-3, steps + 1, device=adapter.device)
    traces: list[StepTrace] = []
    torch.cuda.synchronize(adapter.device)
    started = time.perf_counter()
    for step in range(steps):
        state.step_index = step
        state.timestep_t = timesteps[step]
        state.timestep_s = timesteps[step + 1]
        pre = x.clone()
        output = adapter.forward_state(state, need_hidden=False)
        mask = output.masked_positions
        confidence, proposals = adapter.propose_tokens_and_confidence(
            output, alg, temperature, top_p, top_k
        )
        budget = adapter.compute_commit_budget(mask, timesteps[step], timesteps[step + 1], step == steps - 1)
        full_confidence = torch.full_like(x, -torch.inf, dtype=output.aligned_logits.dtype)
        full_confidence[mask] = confidence
        if budget > 0:
            if alg_temp is None or alg_temp == 0:
                transfer_index = torch.topk(full_confidence, budget, dim=-1).indices
            else:
                probabilities = F.softmax(full_confidence / alg_temp, dim=-1)
                transfer_index = torch.multinomial(probabilities, num_samples=budget)
            proposed_full = torch.full_like(x, adapter.mask_token_id)
            proposed_full[mask] = proposals
            rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(transfer_index)
            x[rows, transfer_index] = proposed_full[rows, transfer_index]
        else:
            transfer_index = torch.empty((x.shape[0], 0), dtype=torch.long, device=x.device)
        state.input_ids = x
        if capture_trace:
            # The official generation hook can only infer the committed set by
            # diffing pre/post states, so it reports positions in ascending
            # tensor order.  Canonicalize the independent trace the same way;
            # top-k rank is not part of the observable generation transition.
            positions = torch.sort(transfer_index[0]).values
            traces.append(
                StepTrace(
                    step=step,
                    pre_state=clone_cpu(pre),
                    post_state=clone_cpu(x),
                    commit_positions=clone_cpu(positions),
                    committed_tokens=clone_cpu(x[0, positions]),
                    proposal_tokens=clone_cpu(proposals),
                    logits_digest=logits_digest(output.aligned_logits, mask),
                )
            )
    torch.cuda.synchronize(adapter.device)
    latency = time.perf_counter() - started
    return GenerationResult(
        sequences=x,
        texts=adapter.decode(x, state.prompt_length),
        traces=traces,
        latency_seconds=latency,
        effective_nfe=steps,
    )


def compare_generation_traces(official: GenerationResult, custom: GenerationResult) -> dict[str, Any]:
    step_rows: list[dict[str, Any]] = []
    passed = torch.equal(official.sequences.cpu(), custom.sequences.cpu())
    if len(official.traces) != len(custom.traces):
        passed = False
    for index, (left, right) in enumerate(zip(official.traces, custom.traces)):
        state_equal = torch.equal(left.post_state, right.post_state)
        positions_equal = torch.equal(left.commit_positions, right.commit_positions)
        tokens_equal = torch.equal(left.committed_tokens, right.committed_tokens)
        proposals_equal = torch.equal(left.proposal_tokens, right.proposal_tokens)
        digest_delta = max(abs(a - b) for a, b in zip(left.logits_digest, right.logits_digest))
        row_pass = (
            state_equal and positions_equal and tokens_equal and proposals_equal and digest_delta <= 1e-6
        )
        passed = passed and row_pass
        step_rows.append(
            {
                "step": index,
                "passed": row_pass,
                "state_equal": state_equal,
                "commit_positions_equal": positions_equal,
                "committed_tokens_equal": tokens_equal,
                "proposal_tokens_equal": proposals_equal,
                "logits_digest_max_abs_diff": digest_delta,
                "commit_count": int(left.commit_positions.numel()),
            }
        )
    return {
        "passed": bool(passed),
        "final_sequences_equal": torch.equal(official.sequences.cpu(), custom.sequences.cpu()),
        "final_text_equal": official.texts == custom.texts,
        "official_nfe": official.effective_nfe,
        "custom_nfe": custom.effective_nfe,
        "steps": step_rows,
    }
