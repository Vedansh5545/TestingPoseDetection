# skeleton_utils.py

import torch
from model import create_edge_index

# Build the exact edge list you trained on:
edge_index = create_edge_index()           # shape: [2, 2*E]
_edges = edge_index.t().cpu().numpy()      # shape: [2*E, 2]
# Keep only unique undirected bones (i < j)
MPIINF_EDGES = [(i, j) for i, j in _edges if i < j]

# === MediaPipe → MPI-INF mapping ===
# Keys: MediaPipe idx 0..32
# Values: MPI-INF joint idx 0..27, or None if unused.
MEDIAPIPE_TO_MPIINF = {i: None for i in range(33)}

# Body landmarks:
MEDIAPIPE_TO_MPIINF[23] = 5   # Left Hip
MEDIAPIPE_TO_MPIINF[24] = 1   # Right Hip
MEDIAPIPE_TO_MPIINF[25] = 6   # Left Knee
MEDIAPIPE_TO_MPIINF[26] = 2   # Right Knee
MEDIAPIPE_TO_MPIINF[27] = 7   # Left Ankle
MEDIAPIPE_TO_MPIINF[28] = 3   # Right Ankle
MEDIAPIPE_TO_MPIINF[29] = 8   # Left Heel
MEDIAPIPE_TO_MPIINF[30] = 4   # Right Heel

MEDIAPIPE_TO_MPIINF[11] = 12  # Left Shoulder
MEDIAPIPE_TO_MPIINF[12] = 15  # Right Shoulder
MEDIAPIPE_TO_MPIINF[13] = 13  # Left Elbow
MEDIAPIPE_TO_MPIINF[14] = 16  # Right Elbow
MEDIAPIPE_TO_MPIINF[15] = 14  # Left Wrist
MEDIAPIPE_TO_MPIINF[16] = 17  # Right Wrist

MEDIAPIPE_TO_MPIINF[0]  = 11  # Nose → Head Top
# Pelvis (0) and Neck (10) will be derived at runtime.
