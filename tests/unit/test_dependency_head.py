import torch

from treepc.models.dependency_head import DependencyHead
from treepc.training.losses import dependency_loss


def test_dependency_head_is_symmetric_nonnegative_and_masks_padding() -> None:
    model = DependencyHead(
        hidden_size=16,
        projection_size=8,
        relative_position_size=6,
        max_relative_position=10,
        pair_hidden_size=16,
        pair_chunk_size=5,
    )
    hidden = torch.randn(2, 4, 16)
    positions = torch.tensor([[2, 4, 7, 9], [1, 3, 5, 8]])
    timestep = torch.tensor([0.8, 0.2])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    weights, raw_scores = model(hidden, positions, timestep, valid, return_raw_scores=True)
    assert weights.shape == (2, 4, 4)
    assert raw_scores.shape == weights.shape
    torch.testing.assert_close(weights, weights.transpose(-1, -2))
    assert torch.all(weights >= 0)
    assert torch.all(weights.diagonal(dim1=-2, dim2=-1) == 0)
    assert torch.all(weights[1, 2:] == 0)
    assert torch.all(weights[1, :, 2:] == 0)
    assert torch.all(raw_scores[1, 2:] == 0)
    assert torch.all(raw_scores[1, :, 2:] == 0)


def test_learned_relative_position_embedding_is_directional_clipped_and_trainable() -> None:
    model = DependencyHead(
        hidden_size=8,
        projection_size=4,
        timestep_size=4,
        relative_position_size=3,
        max_relative_position=5,
        pair_hidden_size=8,
    )
    left = torch.tensor([10, 2, -10])
    right = torch.tensor([7, 7, 10])
    indices = model._relative_position_indices(left, right)
    assert torch.equal(indices, torch.tensor([8, 0, 0]))

    _, raw_scores = model(
        torch.randn(1, 3, 8),
        torch.tensor([[2, 7, 10]]),
        torch.tensor([0.5]),
        return_raw_scores=True,
    )
    raw_scores.sum().backward()
    gradient = model.relative_position_embedding.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_legacy_scalar_distance_checkpoint_is_migrated() -> None:
    model = DependencyHead(
        hidden_size=8,
        projection_size=4,
        timestep_size=4,
        relative_position_size=3,
        pair_hidden_size=8,
    )
    state = model.state_dict()
    base_size = 4 * model.projection_size
    current_input = state["edge_mlp.0.weight"]
    legacy_input = torch.cat(
        [
            current_input[:, :base_size],
            torch.randn(current_input.shape[0], 1),
            current_input[:, -model.timestep_size :],
        ],
        dim=-1,
    )
    legacy_state = {key: value for key, value in state.items() if key != "relative_position_embedding.weight"}
    legacy_state["edge_mlp.0.weight"] = legacy_input

    restored = DependencyHead(
        hidden_size=8,
        projection_size=4,
        timestep_size=4,
        relative_position_size=3,
        pair_hidden_size=8,
    )
    assert restored.load_compatible_state_dict(legacy_state) is True


def test_symmetry_loss_uses_raw_scores_and_produces_gradients() -> None:
    raw_scores = torch.tensor(
        [[[0.0, 0.8, 0.2], [0.4, 0.0, 0.7], [0.5, 0.1, 0.0]]],
        requires_grad=True,
    )
    prediction = 0.5 * (raw_scores + raw_scores.transpose(-1, -2))
    target = prediction.detach()

    loss, components = dependency_loss(
        prediction,
        raw_scores,
        target,
        regression_weight=0.0,
        ranking_weight=0.0,
        symmetry_weight=1.0,
    )
    loss.backward()

    expected = torch.tensor((0.4 + 0.3 + 0.6) / 3)
    torch.testing.assert_close(components["symmetry"], expected)
    torch.testing.assert_close(loss, expected)
    assert raw_scores.grad is not None
    assert raw_scores.grad.abs().sum() > 0
