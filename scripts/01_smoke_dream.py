#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from treepc.dream.adapter import DreamAdapter
from treepc.dream.generation import custom_independent_generate
from treepc.dream.loader import load_dream
from treepc.utils.io import write_json
from treepc.utils.seed import seed_everything

PROMPTS = [
    "Return the number 7.",
    "What is 2 + 3?",
    "Write a Python function that returns True.",
    "Name one primary color.",
    "Complete: one, two, three, ...",
    "What is the capital of France?",
    "Return only the word hello.",
    "Calculate 6 * 8.",
    "Write the Python expression for an empty list.",
    "What comes after Monday?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()
    seed_everything(args.seed)
    loaded = load_dream(args.device)
    adapter = DreamAdapter(loaded)
    rows = []
    try:
        first = adapter.encode_prompt(PROMPTS[0])
        state = adapter.make_initial_state(first, 32, 4)
        with torch.inference_mode():
            no_hidden = adapter.model(
                state.input_ids, state.attention_mask, state.tok_idx,
                output_hidden_states=False, output_attentions=False, return_dict=True,
            ).logits
            with_hidden = adapter.model(
                state.input_ids, state.attention_mask, state.tok_idx,
                output_hidden_states=True, output_attentions=False, return_dict=True,
            ).logits
        hidden_logits_equal = torch.equal(no_hidden, with_hidden)
        initialized_as_mask = bool(
            state.input_ids[:, state.prompt_length :].eq(adapter.mask_token_id).all().item()
        )
        rendered_prompt_valid = PROMPTS[0] in adapter.render_prompt(PROMPTS[0])

        short_prompt, long_prompt = PROMPTS[0], PROMPTS[2]
        standalone = adapter.make_initial_state(adapter.encode_prompt(short_prompt), 8, 2)
        padded = adapter.make_initial_state(adapter.encode_prompts([short_prompt, long_prompt]), 8, 2)
        with torch.inference_mode():
            standalone_logits = adapter.forward_state(standalone).aligned_logits[:, -8:]
            padded_logits = adapter.forward_state(padded).aligned_logits[:1, -8:]
        padding_preserves_generation_argmax = torch.equal(
            standalone_logits.argmax(dim=-1), padded_logits.argmax(dim=-1)
        )
        for prompt in PROMPTS[: args.max_examples]:
            result = custom_independent_generate(
                adapter, prompt, steps=4, max_new_tokens=32, capture_trace=True
            )
            prompt_ids = adapter.encode_prompt(prompt)["input_ids"]
            rows.append(
                {
                    "prompt": prompt,
                    "text": result.texts[0],
                    "prompt_preserved": torch.equal(result.sequences[:, : prompt_ids.shape[1]], prompt_ids),
                    "residual_masks": int(result.sequences.eq(adapter.mask_token_id).sum().item()),
                    "nfe": result.effective_nfe,
                }
            )
        report = {
            "passed": hidden_logits_equal
            and initialized_as_mask
            and rendered_prompt_valid
            and padding_preserves_generation_argmax
            and all(r["prompt_preserved"] and r["residual_masks"] == 0 for r in rows),
            "hidden_request_preserves_logits": hidden_logits_equal,
            "generation_span_initialized_as_mask": initialized_as_mask,
            "chat_template_contains_prompt": rendered_prompt_valid,
            "batch_padding_preserves_generation_argmax": padding_preserves_generation_argmax,
            "sdpa_active": loaded.metadata["attention_classes"] == ["DreamSdpaAttention"],
            "model": loaded.metadata,
            "rows": rows,
        }
        write_json(args.output, report)
        if not report["passed"]:
            raise SystemExit(1)
        print(f"Dream smoke passed: {Path(args.output)}")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
