import pytest
import torch

from treepc.dream.adapter import DreamAdapter
from treepc.dream.generation import compare_generation_traces, custom_independent_generate, official_generate
from treepc.dream.loader import load_dream


@pytest.mark.cuda
@pytest.mark.model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_official_and_custom_four_step_generation_match() -> None:
    adapter = DreamAdapter(load_dream("cuda:0"))
    try:
        prompt = "Return only the number 7."
        official = official_generate(adapter, prompt, steps=4, max_new_tokens=16, capture_trace=True)
        custom = custom_independent_generate(adapter, prompt, steps=4, max_new_tokens=16, capture_trace=True)
        assert compare_generation_traces(official, custom)["passed"]
    finally:
        adapter.close()
