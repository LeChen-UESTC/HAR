from __future__ import annotations

import numpy as np


def edge2mat(link: list[tuple[int, int]], num_node: int) -> np.ndarray:
    adjacency = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        adjacency[j, i] = 1.0
    return adjacency


def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=0)
    node_count = adjacency.shape[0]
    degree_norm = np.zeros((node_count, node_count), dtype=np.float32)
    for i in range(node_count):
        if degree[i] > 0:
            degree_norm[i, i] = degree[i] ** (-1)
    return np.dot(adjacency, degree_norm)


def get_spatial_graph(
    num_node: int,
    self_link: list[tuple[int, int]],
    inward: list[tuple[int, int]],
    outward: list[tuple[int, int]],
) -> np.ndarray:
    identity = edge2mat(self_link, num_node)
    inward_graph = normalize_digraph(edge2mat(inward, num_node))
    outward_graph = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((identity, inward_graph, outward_graph)).astype(np.float32)


NUM_NODE = 25
SELF_LINK = [(i, i) for i in range(NUM_NODE)]
INWARD_ORI_INDEX = [
    (1, 2),
    (2, 21),
    (3, 21),
    (4, 3),
    (5, 21),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 21),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (22, 23),
    (23, 8),
    (24, 25),
    (25, 12),
]
INWARD = [(i - 1, j - 1) for (i, j) in INWARD_ORI_INDEX]
OUTWARD = [(j, i) for (i, j) in INWARD]
NEIGHBOR = INWARD + OUTWARD


class NTUGraph:
    def __init__(self, labeling_mode: str = "spatial") -> None:
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = NUM_NODE
        self.self_link = SELF_LINK
        self.inward = INWARD
        self.outward = OUTWARD
        self.neighbor = NEIGHBOR

    def get_adjacency_matrix(self, labeling_mode: str | None = None) -> np.ndarray:
        if labeling_mode is None:
            return self.A
        if labeling_mode != "spatial":
            raise ValueError(f"Unsupported NTU graph labeling_mode={labeling_mode}")
        return get_spatial_graph(NUM_NODE, SELF_LINK, INWARD, OUTWARD)
