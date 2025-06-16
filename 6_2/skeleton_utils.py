# skeleton_utils.py
import torch
from model import create_edge_index

# 17 COCO joints ← MediaPipe indices
MEDIAPIPE_TO_COCO17 = {
    0:  0,    # nose
    2:  1,  5: 2,   # eyes
    7:  3,  8: 4,   # ears
   11:  5, 12: 6,   # shoulders
   13:  7, 14: 8,   # elbows
   15:  9, 16:10,   # wrists
   23: 11, 24:12,   # hips
   25: 13, 26:14,   # knees
   27: 15, 28:16,   # ankles
}
for i in range(33):
    MEDIAPIPE_TO_COCO17.setdefault(i, None)

# Undirected COCO17 bone list
COCO17_EDGES = [
    (0,1),(0,2),
    (1,3),(2,4),
    (5,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
]

# For PyG GCNConv, build bidirectional edge_index
EDGE_INDEX = torch.tensor(
    COCO17_EDGES + [(j,i) for i,j in COCO17_EDGES],
    dtype=torch.long).t().contiguous()


# Generate the 2×E edge_index from your model definition
edge_index = create_edge_index()                     # shape: [2, E]
_edges = edge_index.t().cpu().numpy()                # shape: [E, 2]
# Keep only the undirected unique edges (i < j)
MPIINF_EDGES = [(i, j) for i, j in _edges if i < j]

# Map each of the 33 MediaPipe landmarks to your 28-joint MPI-INF skeleton
# Keys: MediaPipe index 0..32
# Values: MPI-INF joint index 0..27, or None if unused
MEDIAPIPE_TO_MPIINF = {i: (i if i < 28 else None) for i in range(33)}