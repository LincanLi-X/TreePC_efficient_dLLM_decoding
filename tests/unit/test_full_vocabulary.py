import torch

from treepc.models.correction_head import ConditionalCorrectionHead


def test_full_vocab_equals_selected_projection_and_zero_is_identity():
    torch.manual_seed(1)
    embeddings = torch.randn(37, 8)
    head = ConditionalCorrectionHead(embeddings, embeddings, hidden_size=8, rank=2, feature_size=4)
    logits = torch.randn(2, 37)
    kwargs = dict(
        child_hidden=torch.randn(2, 8),
        parent_hidden=torch.randn(2, 8),
        parent_tokens=torch.tensor([1, 2]),
        relative_position=torch.tensor([1.0, 2.0]),
        sequence_length=torch.tensor([20.0, 20.0]),
        timestep=torch.tensor([0.5, 0.5]),
        dependency_weight=torch.tensor([0.1, 0.2]),
    )
    full, _, _ = head(**kwargs, token_ids=None, base_logits=logits)
    torch.testing.assert_close(full, logits)
    torch.nn.init.normal_(head.up.weight)
    full, _, _ = head(**kwargs, token_ids=None, base_logits=logits)
    ids = torch.tensor([[0, 20, 36], [2, 19, 35]])
    subset, _, _ = head(**kwargs, token_ids=ids, base_logits=logits.gather(-1, ids))
    torch.testing.assert_close(subset, full.gather(-1, ids))
    identity, _, _ = head(**kwargs, token_ids=None, base_logits=logits, force_gate_zero=True)
    torch.testing.assert_close(identity, logits)
