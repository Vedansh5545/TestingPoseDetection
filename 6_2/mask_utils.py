# mask_utils.py

import numpy as np

def joint_visibility_mask(pose28: np.ndarray, thresh: float = 0.1) -> np.ndarray:
    """
    pose28: (28,3) array of [x_px, y_px, confidence]
    Returns a boolean array of shape (28,) where True means
    that joint's confidence > thresh.
    """
    return pose28[:, 2] > thresh

def bone_visibility_mask(pose28: np.ndarray,
                         edges: list[tuple[int,int]],
                         thresh: float = 0.1) -> np.ndarray:
    """
    Returns a boolean array of length len(edges). Each entry is True
    if *both* endpoints of that bone have confidence > thresh.
    """
    jmask = joint_visibility_mask(pose28, thresh)
    return np.array([jmask[i] and jmask[j] for i, j in edges], dtype=bool)
