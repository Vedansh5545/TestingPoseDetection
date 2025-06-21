# skeleton_utils.py

import torch
from model import create_edge_index

# ─── 1) Build the 28-joint undirected edge list ─────────────────────────
# This must match what your model uses in create_edge_index()
edge_index = create_edge_index()           # shape: [2, E]
_edges    = edge_index.t().cpu().numpy()   # shape: [E, 2]
MPIINF_EDGES = [(i, j) for i, j in _edges if i < j]

# ─── 2) MediaPipe → MPI-INF joint mapping ───────────────────────────────
# See MediaPipe Pose landmark indices: https://google.github.io/mediapipe/solutions/pose.html
# We map only the MediaPipe landmarks that correspond to our 28 MPI-INF joints.
# All others are set to None.

MEDIAPIPE_TO_MPIINF = {
    # Head & neck
    0:  11,   # nose → head_top
    11: 12,   # left_shoulder mediapipe idx 11 → MPIINF joint 12
    12: 15,   # right_shoulder → MPIINF joint 15
    # Arms
    13: 13,   # left_elbow → MPIINF 13
    14: 16,   # right_elbow → MPIINF 16
    15: 14,   # left_wrist → MPIINF 14
    16: 17,   # right_wrist → MPIINF 17
    # Hips & legs
    23: 5,    # left_hip → MPIINF 5
    24: 1,    # right_hip → MPIINF 1
    25: 6,    # left_knee → MPIINF 6
    26: 2,    # right_knee → MPIINF 2
    27: 7,    # left_ankle → MPIINF 7
    28: 3,    # right_ankle → MPIINF 3
    29: 8,    # left_heel → MPIINF 8
    30: 4,    # right_heel → MPIINF 4
    # Everything else unmapped
    **{i: None for i in range(33) if i not in {
        0,11,12,13,14,15,16,23,24,25,26,27,28,29,30
    }}
}

# ─── 3) (Optional) Convenience functions ────────────────────────────────

def joint_visibility_mask(pose28: torch.Tensor, thresh: float = 0.1):
    """
    Returns a boolean mask of shape (28,) indicating which joints
    have confidence > thresh.
    """
    # pose28 can be numpy or torch Tensor with shape (28,3)
    conf = pose28[:, 2]
    mask = (conf > thresh)
    return mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

def bone_visibility_mask(pose28: torch.Tensor, thresh: float = 0.1):
    """
    Returns a boolean list of length E, one per MPIINF_EDGES,
    indicating which bones have both endpoints visible.
    """
    jmask = joint_visibility_mask(pose28, thresh)
    return [jmask[i] and jmask[j] for i, j in MPIINF_EDGES]
