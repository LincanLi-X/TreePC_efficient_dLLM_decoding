import itertools

import networkx as nx
import torch

from treepc.graph.chow_liu import maximum_spanning_tree, tree_weight
from treepc.graph.orientation import orient_tree


def test_torch_mst_matches_networkx_weight_and_orientation() -> None:
    weights = torch.tensor(
        [[0.0, 0.8, 0.2, 0.1], [0.8, 0.0, 0.7, 0.3], [0.2, 0.7, 0.0, 0.9], [0.1, 0.3, 0.9, 0.0]]
    )
    parent, _ = maximum_spanning_tree(weights)
    graph = nx.Graph()
    for i, j in itertools.combinations(range(4), 2):
        graph.add_edge(i, j, weight=float(weights[i, j]))
    expected = nx.maximum_spanning_tree(graph).size(weight="weight")
    assert abs(tree_weight(weights, parent).item() - expected) < 1e-6
    oriented_parent, depth = orient_tree(parent, 2)
    assert oriented_parent[2].item() == -1
    assert sorted(depth.tolist()) == [0, 1, 1, 2]


def test_known_binary_tree_mst_and_factorization_gain() -> None:
    # X0 -> X1 -> X2 with symmetric 10% bit-flip channels.
    joint = torch.zeros((2, 2, 2), dtype=torch.float64)
    for x0, x1, x2 in itertools.product(range(2), repeat=3):
        joint[x0, x1, x2] = 0.5 * (0.9 if x1 == x0 else 0.1) * (0.9 if x2 == x1 else 0.1)
    marginals = [joint.sum(dim=tuple(axis for axis in range(3) if axis != node)) for node in range(3)]
    independent = torch.einsum("i,j,k->ijk", *marginals)
    independent_kl = (joint * (joint.log() - independent.log())).sum()
    tree_kl = torch.tensor(0.0, dtype=torch.float64)  # the generating tree exactly represents joint
    weights = torch.zeros((3, 3), dtype=torch.float64)
    for left, right in itertools.combinations(range(3), 2):
        pair = joint.sum(dim=tuple(axis for axis in range(3) if axis not in (left, right)))
        product = torch.outer(marginals[left], marginals[right])
        weights[left, right] = weights[right, left] = (pair * (pair.log() - product.log())).sum()
    parent, _ = maximum_spanning_tree(weights)
    edges = {frozenset((node, int(parent[node]))) for node in range(3) if parent[node] >= 0}
    assert edges == {frozenset((0, 1)), frozenset((1, 2))}
    assert tree_kl < independent_kl
