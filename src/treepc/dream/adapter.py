from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from treepc.dream.alignment import align_dream_hidden, align_dream_logits
from treepc.dream.loader import LoadedDream
from treepc.types import DreamState, DreamStepOutput


def top_p_logits(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: Tensor, top_k: int) -> Tensor:
    top_k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def max_token_probability(logits: Tensor) -> Tensor:
    """Return max_v p(v) for each position under the untruncated posterior."""
    return torch.softmax(logits.float(), dim=-1).amax(dim=-1)


def sample_tokens(
    logits: Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    alg: str = "maskgit_plus",
) -> tuple[Tensor, Tensor]:
    """Exact local equivalent of Dream's official sample_tokens semantics."""
    work = logits / temperature if temperature > 0 else logits
    if top_p is not None and top_p < 1:
        work = top_p_logits(work, top_p)
    if top_k is not None:
        work = top_k_logits(work, top_k)
    probs = F.softmax(work, dim=-1)
    if temperature > 0:
        token_ids = torch.distributions.Categorical(probs=probs).sample()
        confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, token_ids = probs.max(dim=-1)
    if alg == "topk_margin":
        top_two = torch.topk(probs, 2, dim=-1).values
        confidence = top_two[..., 0] - top_two[..., 1]
    elif alg == "entropy":
        confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    elif alg not in {"origin", "maskgit_plus"}:
        raise ValueError(f"Unknown Dream algorithm: {alg}")
    return confidence, token_ids


class DreamAdapter:
    def __init__(self, loaded: LoadedDream) -> None:
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.device = loaded.device
        self.dtype = loaded.dtype
        self.metadata = loaded.metadata
        self.mask_token_id = int(self.model.config.mask_token_id)

    def render_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def encode_prompt(self, prompt: str) -> dict[str, Tensor]:
        return self.encode_prompts([prompt])

    def encode_prompts(self, prompts: list[str]) -> dict[str, Tensor]:
        encoded = self.tokenizer(
            [self.render_prompt(prompt) for prompt in prompts],
            return_tensors="pt",
            add_special_tokens=False,
            padding=len(prompts) > 1,
        )
        return {key: value.to(self.device) for key, value in encoded.items() if isinstance(value, Tensor)}

    def make_initial_state(self, encoded: dict[str, Tensor], max_new_tokens: int, steps: int) -> DreamState:
        input_ids = encoded["input_ids"]
        prompt_length = input_ids.shape[1]
        x = F.pad(input_ids, (0, max_new_tokens), value=self.mask_token_id)
        attention_mask, tok_idx = self.prepare_attention(encoded.get("attention_mask"), max_new_tokens)
        generation_mask = torch.zeros_like(x, dtype=torch.bool)
        generation_mask[:, prompt_length:] = True
        timesteps = torch.linspace(1, 1e-3, steps + 1, device=self.device)
        return DreamState(
            input_ids=x,
            attention_mask=attention_mask,
            tok_idx=tok_idx,
            prompt_length=prompt_length,
            generation_mask=generation_mask,
            step_index=0,
            timestep_t=timesteps[0],
            timestep_s=timesteps[1],
        )

    def prepare_attention(
        self, attention_mask: Tensor | None, max_new_tokens: int
    ) -> tuple[Tensor | str, Tensor | None]:
        if attention_mask is not None and torch.any(attention_mask == 0):
            padded = F.pad(attention_mask, (0, max_new_tokens), value=1.0)
            tok_idx = padded.long().cumsum(-1) - 1
            tok_idx.masked_fill_(padded == 0, 1)
            pair_mask = torch.logical_and(
                padded.unsqueeze(1).unsqueeze(-2), padded.unsqueeze(1).unsqueeze(-1)
            )
            return pair_mask, tok_idx
        return "full", None

    def forward_state(self, state: DreamState, need_hidden: bool = False) -> DreamStepOutput:
        output = self.model(
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
            position_ids=state.tok_idx,
            use_cache=False,
            output_hidden_states=need_hidden,
            output_attentions=False,
            return_dict=True,
        )
        aligned_logits = align_dream_logits(output.logits)
        aligned_hidden = align_dream_hidden(output.hidden_states[-1]) if need_hidden else None
        masked = state.input_ids.eq(self.mask_token_id) & state.generation_mask
        return DreamStepOutput(
            aligned_logits=aligned_logits,
            aligned_hidden=aligned_hidden,
            masked_positions=masked,
            base_sampled_tokens=torch.empty(0, dtype=torch.long, device=self.device),
            confidence=torch.empty(0, dtype=aligned_logits.dtype, device=self.device),
        )

    def propose_tokens_and_confidence(
        self,
        output: DreamStepOutput,
        alg: str,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
    ) -> tuple[Tensor, Tensor]:
        mask_logits = output.aligned_logits[output.masked_positions]
        confidence, tokens = sample_tokens(mask_logits, temperature, top_p, top_k, alg)
        output.confidence = confidence
        output.base_sampled_tokens = tokens
        return confidence, tokens

    @staticmethod
    def compute_commit_budget(masked: Tensor, t: Tensor, s: Tensor, is_last: bool) -> int:
        num_mask_token = masked.sum() / masked.shape[0]
        return int(num_mask_token) if is_last else int(num_mask_token * (1 - s / t))

    def decode(self, sequences: Tensor, prompt_length: int) -> list[str]:
        texts: list[str] = []
        stop_ids = {self.mask_token_id, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
        for row in sequences[:, prompt_length:].tolist():
            kept: list[int] = []
            for token in row:
                if token in stop_ids:
                    break
                kept.append(token)
            texts.append(self.tokenizer.decode(kept, skip_special_tokens=True).strip())
        return texts

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()


def logits_digest(logits: Tensor, mask: Tensor) -> list[float]:
    values = logits[mask].float()
    if values.numel() == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [values.mean().item(), values.std().item(), values.min().item(), values.max().item()]
