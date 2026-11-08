"""Anatomical graph matching the saved CARE-PD SMPL9 joint axis.

Coordinates have shape (T, 9, 3). Graph IDs are saved-array indices,
not native SMPL indices. No coordinates or dataset files are needed here.
"""

from collections import deque

import numpy as np


JOINT_NAMES = (
    "pelvis",       # 0
    "left_hip",     # 1
    "left_knee",    # 2
    "left_ankle",   # 3
    "left_foot",    # 4
    "right_hip",    # 5
    "right_knee",   # 6
    "right_ankle",  # 7
    "right_foot",   # 8
)
SMPL_INDICES = (0, 1, 4, 7, 10, 2, 5, 8, 11) 

NUM_NODES = len(JOINT_NAMES)
LEFT_FOOT = 4
# Anatomical connections within one frame.
EDGES = (
    (0, 1),  # Pelvis — left hip
    (1, 2),  # Left hip — left knee
    (2, 3),  # Left knee — left ankle
    (3, 4),  # Left ankle — left foot
    (0, 5),  # Pelvis — right hip
    (5, 6),  # Right hip — right knee
    (6, 7),  # Right knee — right ankle
    (7, 8),  # Right ankle — right foot
)


# def build_anatomical_adjacency():
#     """Return binary undirected (9, 9) adjacency WITHOUT self-loops."""
#     adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
#     for i, j in EDGES:
#         adjacency[i, j] = 1.0
#         adjacency[j, i] = 1.0
#     return adjacency


def shortest_path_distances():
    """Return integer hop distances; distance[i, i] is zero.

    Breadth-first search counts anatomical edges, not Euclidean distances
    or the number of possible walks between nodes.
    """
    neighbors = [[] for _ in range(NUM_NODES)]
    for i, j in EDGES:
        neighbors[i].append(j)
        neighbors[j].append(i)

    distance = np.full((NUM_NODES, NUM_NODES), -1, dtype=np.int64)
    for root in range(NUM_NODES):
        distance[root, root] = 0
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if distance[root, neighbor] == -1:
                    distance[root, neighbor] = distance[root, current] + 1
                    queue.append(neighbor)

    if np.any(distance < 0):
        raise ValueError("SMPL9 anatomical graph must be connected")
    return distance
