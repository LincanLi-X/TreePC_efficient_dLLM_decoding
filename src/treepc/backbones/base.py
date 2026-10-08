"""Shared adapter contract; schedulers retain their backbone-native semantics."""

from typing import Protocol

from torch import Tensor


class BackboneAdapter(Protocol):
    model: object
    device: object
    mask_token_id: int
    forward_count: int

    def encode_prompt(self, prompt: str) -> dict[str, Tensor]: ...
    def make_initial_state(self, encoded, max_new_tokens, steps): ...
    def forward_state(self, state, need_hidden=False): ...
    def propose_tokens_and_confidence(self, output, alg, temperature, top_p, top_k): ...
    def compute_commit_budget(self, masked, t, s, is_last): ...
    def sample_corrected(self, logits, temperature, top_p, top_k): ...


def load_adapter(device="auto", config_path=None):
    from treepc.dream.adapter import DreamAdapter
    from treepc.dream.loader import load_dream

    return DreamAdapter(load_dream(device, config_path))
