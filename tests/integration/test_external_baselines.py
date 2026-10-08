"""External API smoke checks; vanilla LLaDA tests CAD only, not DSCD quality."""

import os
from pathlib import Path

import pytest
import torch

from treepc.backbones.base import load_adapter
from treepc.dream.generation import custom_independent_generate
from treepc.evaluation.external import ExternalDecoder


@pytest.mark.model
@pytest.mark.cuda
@pytest.mark.parametrize(
    "backbone,method", [("dream", "fast_dllm"), ("llada", "fast_dllm"), ("llada", "cd4lm")]
)
def test_official_external_decoder(backbone, method, monkeypatch):
    root = os.environ.get("TREEPC_TEST_MODEL_ROOT")
    checkout = os.environ.get("TREEPC_TEST_FAST_ROOT" if method == "fast_dllm" else "TREEPC_TEST_CD4LM_ROOT")
    if not root or not checkout:
        pytest.skip("Provide external checkpoint and official source roots")
    monkeypatch.setenv("TREEPC_BACKBONE", backbone)
    monkeypatch.setenv(
        "TREEPC_MODEL_DIR", str(Path(root) / ("DREAM-7B" if backbone == "dream" else "LLaDA-8B"))
    )
    adapter = load_adapter("cuda:0")
    try:
        decoder = ExternalDecoder(adapter, method, checkout, 0.9)
        calls = []
        hook = adapter.model.register_forward_pre_hook(lambda module, inputs: calls.append(1))
        try:
            sequence = decoder("What is 2+2?", 4, 16)
            assert sequence.ndim == 2 and calls
            assert not sequence[0, -16:].eq(adapter.mask_token_id).any()
            if backbone == "llada" and method == "fast_dllm":
                prompt = "What is 2+2?"
                official, _ = decoder.generate(
                    adapter.model,
                    adapter.encode_prompt(prompt)["input_ids"],
                    steps=4,
                    gen_length=16,
                    block_length=16,
                    threshold=None,
                )
                native = custom_independent_generate(adapter, prompt, steps=4, max_new_tokens=16)
                assert torch.equal(official, native.sequences)
        finally:
            hook.remove()
    finally:
        adapter.close()
