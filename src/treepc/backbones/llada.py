"""LLaDA full-block, CFG=0, low-confidence remasking.

Reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
No causal next-token shift, no KV cache, no premature stopping.
"""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from treepc.dream.adapter import DreamAdapter, top_k_logits, top_p_logits
from treepc.dream.loader import LoadedDream, resolve_device, resolve_model_dir
from treepc.types import DreamStepOutput
from treepc.utils.io import sha256_file


def load_llada(device, config, model_revision=None):
    path = resolve_model_dir(config)
    if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"Incomplete LLaDA checkpoint: {path}")
    device = resolve_device(device, float(config.get("minimum_free_memory_gib", 24)))
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            revision=model_revision,
        )
        .to(device)
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedDream(
        model,
        tokenizer,
        device,
        torch.bfloat16,
        {
            "backbone": "llada",
            "repo_id": config["repo_id"],
            "local_dir": str(Path(path)),
            "device": str(device),
            "dtype": "torch.bfloat16",
            "mask_token_id": model.config.mask_token_id,
            "gpu_name": torch.cuda.get_device_name(device),
            "model_revision": model_revision,
            "scheduler": "full_block_low_confidence",
            "cfg_scale": 0,
            "asset_sha256": {p.name: sha256_file(p) for p in Path(path).glob("*.py")},
        },
    )


class LLaDAAdapter(DreamAdapter):
    def prepare_attention(self, attention_mask, max_new_tokens):
        if attention_mask is None:
            return None, None
        return torch.nn.functional.pad(attention_mask, (0, max_new_tokens), value=1), None

    def make_initial_state(self, encoded, max_new_tokens, steps):
        state = super().make_initial_state(encoded, max_new_tokens, steps)
        # Official linear schedule: remainder is committed in the earliest steps.
        self.transfer_counts = [
            max_new_tokens // steps + int(i < max_new_tokens % steps) for i in range(steps)
        ]
        self.budget_step = 0
        return state

    def compute_commit_budget(self, masked, t, s, is_last):
        budget = self.transfer_counts[self.budget_step]
        self.budget_step += 1
        return min(budget, int(masked[0].sum()))

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward_state(self, state, need_hidden=False):
        self.forward_count += 1
        out = self.model(
            input_ids=state.input_ids,
            attention_mask=state.attention_mask if isinstance(state.attention_mask, torch.Tensor) else None,
            use_cache=False,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        return DreamStepOutput(
            aligned_logits=out.logits,
            aligned_hidden=out.hidden_states[-1] if need_hidden else None,
            masked_positions=state.input_ids.eq(self.mask_token_id) & state.generation_mask,
            base_sampled_tokens=torch.empty(0, dtype=torch.long, device=self.device),
            confidence=torch.empty(0, device=self.device),
        )

    def sample_corrected(self, logits, temperature, top_p, top_k):
        work = logits.float()
        if top_p is not None and top_p < 1:
            work = top_p_logits(work, top_p)
        if top_k is not None:
            work = top_k_logits(work, top_k)
        if temperature > 0:
            noise = (-torch.log(torch.rand_like(work, dtype=torch.float64))) ** temperature
            tokens = (work.double().exp() / noise).argmax(-1)
        else:
            tokens = work.argmax(-1)
        confidence = torch.softmax(work, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return confidence, tokens

    def propose_tokens_and_confidence(self, output, alg, temperature, top_p, top_k):
        confidence, tokens = self.sample_corrected(
            output.aligned_logits[output.masked_positions],
            temperature,
            top_p,
            top_k,
        )
        output.confidence, output.base_sampled_tokens = confidence, tokens
        return confidence, tokens
