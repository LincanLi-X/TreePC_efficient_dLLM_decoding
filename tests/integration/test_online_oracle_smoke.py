import pytest
import torch

from treepc.decoding.oracle_treepc import online_oracle_generate
from treepc.dream.adapter import DreamAdapter
from treepc.dream.loader import load_dream


@pytest.mark.cuda
@pytest.mark.model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_online_oracle_finishes_without_masks() -> None:
    adapter = DreamAdapter(load_dream("cuda:0"))
    try:
        result, trace, extra_forwards = online_oracle_generate(
            adapter, "Return only the number 7.", steps=4, max_new_tokens=16, seed=2026
        )
        assert result.effective_nfe == 4
        assert len(trace) == 4
        assert extra_forwards > 0
        assert not result.sequences.eq(adapter.mask_token_id).any()
    finally:
        adapter.close()
