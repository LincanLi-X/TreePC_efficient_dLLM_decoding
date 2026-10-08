import torch

from treepc.models.correction_head import ConditionalCorrectionHead


def test_zero_gate_exactly_recovers_base_logits() -> None:
    embeddings = torch.randn(32, 16)
    model = ConditionalCorrectionHead(embeddings, embeddings.clone(), hidden_size=16, rank=4, feature_size=8)
    batch, top_k = 3, 5
    base = torch.randn(batch, top_k)
    corrected, gate, effective_correction = model(
        child_hidden=torch.randn(batch, 16),
        parent_hidden=torch.randn(batch, 16),
        parent_tokens=torch.randint(0, 32, (batch,)),
        relative_position=torch.tensor([1.0, -2.0, 3.0]),
        sequence_length=torch.tensor([64.0, 64.0, 64.0]),
        timestep=torch.tensor([0.9, 0.5, 0.1]),
        dependency_weight=torch.rand(batch),
        token_ids=torch.randint(0, 32, (batch, top_k)),
        base_logits=base,
        force_gate_zero=True,
    )
    torch.testing.assert_close(corrected, base)
    assert torch.all(gate == 0)
    assert torch.all(effective_correction == 0)


def test_global_scale_is_bounded_and_multiplies_the_residual() -> None:
    torch.manual_seed(7)
    embeddings = torch.randn(32, 16)
    model = ConditionalCorrectionHead(
        embeddings,
        embeddings.clone(),
        hidden_size=16,
        rank=4,
        feature_size=8,
        global_scale_init=0.05,
    )
    torch.nn.init.normal_(model.up.weight, std=0.1)
    inputs = {
        "child_hidden": torch.randn(3, 16),
        "parent_hidden": torch.randn(3, 16),
        "parent_tokens": torch.randint(0, 32, (3,)),
        "relative_position": torch.tensor([1.0, -2.0, 3.0]),
        "sequence_length": torch.tensor([64.0, 128.0, 256.0]),
        "timestep": torch.tensor([0.9, 0.5, 0.1]),
        "dependency_weight": torch.rand(3),
        "token_ids": torch.randint(0, 32, (3, 5)),
        "base_logits": torch.randn(3, 5),
    }
    assert abs(model.global_scale_value() - 0.05) < 1e-6
    with torch.no_grad():
        model.global_scale_logit.copy_(torch.logit(torch.tensor(0.1)))
        corrected_small, _, effective_correction_small = model(**inputs)
        model.global_scale_logit.copy_(torch.logit(torch.tensor(0.2)))
        corrected_large, _, effective_correction_large = model(**inputs)
    small_delta = corrected_small - inputs["base_logits"]
    large_delta = corrected_large - inputs["base_logits"]
    torch.testing.assert_close(effective_correction_small, small_delta)
    torch.testing.assert_close(effective_correction_large, large_delta)
    torch.testing.assert_close(large_delta, 2.0 * small_delta, rtol=1e-4, atol=1e-6)
    assert 0.0 < model.global_scale_value() < 1.0


def test_global_scale_learns_in_tiny_optimization() -> None:
    torch.manual_seed(11)
    embeddings = torch.randn(24, 12)
    model = ConditionalCorrectionHead(
        embeddings,
        embeddings.clone(),
        hidden_size=12,
        rank=4,
        feature_size=8,
        global_scale_init=0.05,
    )
    torch.nn.init.normal_(model.up.weight, std=0.2)
    inputs = {
        "child_hidden": torch.randn(4, 12),
        "parent_hidden": torch.randn(4, 12),
        "parent_tokens": torch.randint(0, 24, (4,)),
        "relative_position": torch.tensor([1.0, -2.0, 3.0, -4.0]),
        "sequence_length": torch.tensor([64.0, 64.0, 64.0, 64.0]),
        "timestep": torch.tensor([0.9, 0.7, 0.4, 0.1]),
        "dependency_weight": torch.rand(4),
        "token_ids": torch.randint(0, 24, (4, 6)),
        "base_logits": torch.randn(4, 6),
    }
    with torch.no_grad():
        model.global_scale_logit.copy_(torch.logit(torch.tensor(0.35)))
        target, _, _ = model(**inputs)
        model.global_scale_logit.copy_(torch.logit(torch.tensor(0.05)))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "global_scale_logit")
    optimizer = torch.optim.Adam([model.global_scale_logit], lr=0.2)
    initial_scale = model.global_scale_value()
    initial_loss = None
    for _ in range(40):
        prediction, _, _ = model(**inputs)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert model.global_scale_value() > initial_scale
    assert float(loss.detach()) < initial_loss * 0.05


def test_legacy_checkpoint_uses_configured_global_scale() -> None:
    embeddings = torch.randn(16, 8)
    source = ConditionalCorrectionHead(embeddings, embeddings.clone(), hidden_size=8, rank=2, feature_size=4)
    legacy_state = {key: value for key, value in source.state_dict().items() if key != "global_scale_logit"}
    restored = ConditionalCorrectionHead(
        embeddings, embeddings.clone(), hidden_size=8, rank=2, feature_size=4
    )
    assert restored.load_compatible_state_dict(legacy_state) is True
    assert abs(restored.global_scale_value() - 0.05) < 1e-6


def test_gate_normalizes_distance_by_each_samples_actual_sequence_length() -> None:
    embeddings = torch.randn(12, 8)
    model = ConditionalCorrectionHead(embeddings, embeddings.clone(), hidden_size=8, rank=2, feature_size=4)
    with torch.no_grad():
        for parameter in model.gate_mlp.parameters():
            parameter.zero_()
        model.gate_mlp[0].weight[0, 2] = 1.0
        model.gate_mlp[2].weight[0, 0] = 1.0
    common = {
        "child_hidden": torch.randn(2, 8),
        "parent_hidden": torch.randn(2, 8),
        "parent_tokens": torch.randint(0, 12, (2,)),
        "relative_position": torch.tensor([50.0, 50.0]),
        "sequence_length": torch.tensor([100.0, 200.0]),
        "timestep": torch.tensor([0.5, 0.5]),
        "dependency_weight": torch.tensor([0.5, 0.5]),
        "token_ids": torch.randint(0, 12, (2, 3)),
        "base_logits": torch.randn(2, 3),
    }

    _, gate, _ = model(**common)

    expected = torch.sigmoid(torch.nn.functional.gelu(torch.tensor([0.5, 0.25])))
    torch.testing.assert_close(gate, expected)
    assert gate[0] > gate[1]
