"""CST-GCN graph supports from Tian et al., equations (3), (4), (6), (7).

Convention: rows are receiving/root nodes; columns are source neighbors.
Temporal columns are ordered oldest frame first, then ascending joint ID.
This module constructs fixed graph supports, not a neural-network layer.
"""

from numbers import Integral

import numpy as np

from .smpl9 import LEFT_FOOT, NUM_NODES, shortest_path_distances


def _validate_radius(radius, distance):
    diameter = int(distance.max())
    if isinstance(radius, bool) or not isinstance(radius, Integral):
        raise ValueError("Spatial radius D must be an integer")
    if not 0 <= radius <= diameter:
        raise ValueError(f"Spatial radius D must be between 0 and {diameter}")


def _validate_window(window_size):
    if (isinstance(window_size, bool)
            or not isinstance(window_size, Integral)
            or window_size < 1 or window_size % 2 != 1):
        raise ValueError("Temporal window Gamma must be a positive odd integer")


def spatial_labels(distance, radius):
    """Return labels[root, neighbor], with -1 outside radius D.

    Equation (3): label depends on BOTH root and neighbor.
    The foot reference is saved node 4, not native SMPL joint 4.
    """
    _validate_radius(radius, distance)
    n = distance.shape[0]
    labels = np.full((n, n), -1, dtype=np.int64)
    for root in range(n):
        for neighbor in range(n):
            hop = int(distance[root, neighbor])
            if hop > radius:
                continue
            closer_to_left_foot = (
                distance[neighbor, LEFT_FOOT] < distance[root, LEFT_FOOT]
            )
            labels[root, neighbor] = (
                2 * hop - 1 if closer_to_left_foot else 2 * hop
            )
    return labels


def build_spatial_adjacency(labels, radius):
    """Equation (4): binary stack of shape (2D+1, N, N)."""
    k = np.arange(2 * radius + 1)[:, None, None]
    return (labels[None, :, :] == k).astype(np.float32)


def cross_spacetime_labels(labels, radius, window_size):
    """Equation (6): labels[root, temporal_slot*N + neighbor].

    For Gamma=3, slots 0/1/2 mean offsets -1/0/+1 from the
    output frame. Outside-radius entries remain -1 at every slot.
    """
    _validate_window(window_size)
    spatial_subsets = 2 * radius + 1
    blocks = []
    for slot in range(window_size):
        blocks.append(np.where(
            labels >= 0, labels + slot * spatial_subsets, -1,
        ))
    return np.concatenate(blocks, axis=1)


def build_cross_spacetime_adjacency(labels_st, radius, window_size):
    """Equation (7): binary stack of shape (Gamma*(2D+1), N, Gamma*N).

    A[k, root, column] is one iff that source has subset label k.
    Never symmetrize these rectangular, directed aggregation supports.
    """
    num_subsets = window_size * (2 * radius + 1)
    k = np.arange(num_subsets)[:, None, None]
    return (labels_st[None, :, :] == k).astype(np.float32)


def normalize_rows(adjacency):
    """Paper's per-subset normalization: D_k^(-1) A_k.

    Works for square or rectangular stacks. Empty subset rows remain
    zero. The paper's normalization is not symmetric Kipf normalization.
    """
    degree = adjacency.sum(axis=-1, keepdims=True)
    normalized = np.zeros_like(adjacency, dtype=np.float32)
    np.divide(adjacency, degree, out=normalized, where=degree != 0)
    return normalized


class CSTGraph:
    """Build SMPL9 graph supports once, for reuse across all walks.

    D=1 and Gamma=3 are explanatory defaults, not claimed paper settings.
    A_s/A_st are binary supports; A_s_normalized/A_st_normalized are the
    supports for the spatial and CST operators respectively.
    """

    def __init__(self, spatial_radius=1, temporal_window=3):
        self.distance = shortest_path_distances()
        _validate_radius(spatial_radius, self.distance)
        _validate_window(temporal_window)
        self.spatial_radius = int(spatial_radius)
        self.temporal_window = int(temporal_window)
        self.num_nodes = NUM_NODES
        self.num_subsets = self.temporal_window * (2 * self.spatial_radius + 1)
        half = self.temporal_window // 2
        self.temporal_offsets = tuple(range(-half, half + 1))

        self.labels_s = spatial_labels(self.distance, self.spatial_radius)
        self.A_s = build_spatial_adjacency(self.labels_s, self.spatial_radius)
        self.A_s_normalized = normalize_rows(self.A_s)

        self.labels_st = cross_spacetime_labels(
            self.labels_s, self.spatial_radius, self.temporal_window,
        )
        self.A_st = build_cross_spacetime_adjacency(
            self.labels_st, self.spatial_radius, self.temporal_window,
        )
        self.A_st_normalized = normalize_rows(self.A_st)


if __name__ == "__main__":
    graph = CSTGraph(spatial_radius=1, temporal_window=3)
    print("Hop distances:\n", graph.distance)
    print("Spatial labels for root 2 (left knee):", graph.labels_s[2])
    print("Temporal offsets:", graph.temporal_offsets)
    print("Spatial adjacency shape:", graph.A_s.shape)
    print("Cross-spacetime adjacency shape:", graph.A_st.shape)
    print("Nonzero columns for root 2 in each CST subset:")
    for k in range(graph.num_subsets):
        columns = np.flatnonzero(graph.A_st[k, 2])
        print(f"  subset {k}: {columns.tolist()}")
