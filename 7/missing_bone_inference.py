# missing_bone_inference.py

import numpy as np

# 1) Define the left↔right symmetry pairs (MPI-INF indices)
SYMMETRY_PAIRS = {
    1: 5,    # right_hip  ↔ left_hip
    2: 6,    # right_knee ↔ left_knee
    3: 7,    # right_ankle↔ left_ankle
    4: 8,    # right_heel ↔ left_heel
    13:16,   # left_elbow ↔ right_elbow
    14:17,   # left_wrist ↔ right_wrist
    12:15,   # left_shoulder ↔ right_shoulder
}

# 2) Precomputed average bone lengths for each edge (i,j)
#    You should compute these once over your training set.
#    Here’s a dummy example—you’ll need to replace with real numbers.
AVERAGE_BONE_LENGTH = {
    (1,2): 150,  (2,3): 150,  # right thigh & shin in mm
    (5,6): 150,  (6,7): 150,  # left thigh & shin
    (12,13): 300, (13,14): 250, # left arm bones
    (15,16): 300, (16,17): 250, # right arm bones
    # …and so on for every bone in MPIINF_EDGES…
}

def infer_missing_joints(pose28: np.ndarray,
                         jmask: np.ndarray) -> np.ndarray:
    """
    Fill in missing joints in `pose28` (28×3) using:
      • symmetry mirror across the pelvis
      • average bone‐length for any remaining gaps.

    Args:
        pose28: np.array of shape (28,3) [x, y, conf]
        jmask:   np.array of shape (28,) bool

    Returns:
        pose28_filled: same shape, with zeros replaced.
    """
    filled = pose28.copy()
    # 1) Mirror via symmetry
    #    Mirror across the pelvis (joint 0)
    pelvis = filled[0,:2]
    for a, b in SYMMETRY_PAIRS.items():
        if not jmask[a] and jmask[b]:
            # p_b relative to pelvis, then mirror sign
            vec = filled[b,:2] - pelvis
            filled[a,:2] = pelvis - vec
            filled[a,2]   = filled[b,2] * 0.8  # slightly lower confidence

        if not jmask[b] and jmask[a]:
            vec = filled[a,:2] - pelvis
            filled[b,:2] = pelvis - vec
            filled[b,2]   = filled[a,2] * 0.8

    # update mask
    new_mask = jmask | (filled[:,2] > 0)

    # 2) For any still‐missing joints, use bone‐length along a known direction
    for (i, j), length in AVERAGE_BONE_LENGTH.items():
        # if joint j is present but i is still missing
        if new_mask[j] and not new_mask[i]:
            direction = filled[j,:2] - filled[0,:2]  # vector from pelvis
            if np.linalg.norm(direction) < 1e-3:
                continue
            direction = direction / np.linalg.norm(direction)
            filled[i,:2] = filled[0,:2] + direction * length
            filled[i,2]   = 0.5
            new_mask[i]   = True

        # same for i present but j missing
        if new_mask[i] and not new_mask[j]:
            direction = filled[i,:2] - filled[0,:2]
            if np.linalg.norm(direction) < 1e-3:
                continue
            direction = direction / np.linalg.norm(direction)
            filled[j,:2] = filled[0,:2] + direction * length
            filled[j,2]   = 0.5
            new_mask[j]   = True

    return filled
