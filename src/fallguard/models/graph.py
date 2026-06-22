from __future__ import annotations

from typing import cast

import numpy as np


def _normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=0)
    inverse = np.zeros_like(adjacency)
    for index, value in enumerate(degree):
        if value > 0:
            inverse[index, index] = value**-1
    return cast(np.ndarray, adjacency @ inverse)


def _hop_distance(num_nodes: int, edges: list[tuple[int, int]], max_hop: int) -> np.ndarray:
    adjacency = np.eye(num_nodes)
    for left, right in edges:
        adjacency[left, right] = 1
        adjacency[right, left] = 1
    distance = np.full((num_nodes, num_nodes), np.inf)
    reachability = np.stack(
        [np.linalg.matrix_power(adjacency, hop) > 0 for hop in range(max_hop + 1)]
    )
    for hop in range(max_hop, -1, -1):
        distance[reachability[hop]] = hop
    return distance


class CocoGraph:
    """COCO-17 ST-GCN spatial graph compatible with MMAction2."""

    num_nodes = 17
    center = 0
    inward = [
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 5),
        (12, 6),
        (9, 7),
        (7, 5),
        (10, 8),
        (8, 6),
        (5, 0),
        (6, 0),
        (1, 0),
        (3, 1),
        (2, 0),
        (4, 2),
    ]

    def __init__(self, max_hop: int = 1) -> None:
        self.max_hop = max_hop
        self.hop_distance = _hop_distance(self.num_nodes, self.inward, max_hop)
        self.A = self._stgcn_spatial()

    def _stgcn_spatial(self) -> np.ndarray:
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        adjacency[self.hop_distance <= self.max_hop] = 1
        normalized = _normalize_digraph(adjacency)
        partitions: list[np.ndarray] = []
        for hop in range(self.max_hop + 1):
            close = np.zeros_like(normalized)
            further = np.zeros_like(normalized)
            for source in range(self.num_nodes):
                for target in range(self.num_nodes):
                    if self.hop_distance[target, source] != hop:
                        continue
                    if (
                        self.hop_distance[target, self.center]
                        >= self.hop_distance[source, self.center]
                    ):
                        close[target, source] = normalized[target, source]
                    else:
                        further[target, source] = normalized[target, source]
            partitions.append(close)
            if hop > 0:
                partitions.append(further)
        return np.stack(partitions).astype(np.float32)
