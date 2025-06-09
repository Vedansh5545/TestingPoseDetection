import torch
import numpy as np
import matplotlib.pyplot as plt
from model_sdg_rat import SDGRATModel

# === Load Trained Model ===
model = SDGRATModel(edge_count=34).to('cuda')
model.load_state_dict(torch.load("final_sdg_rat_model.pth"))
model.eval()

# === Load and Preprocess Sample 2D Pose ===
data = np.load("mpi_inf_combined.npz")
pose2d = data['pose2d'][0]  # First sample
if pose2d.shape[-1] == 2:  # Add confidence if missing
    conf = np.ones_like(pose2d[..., :1])
    pose2d = np.concatenate([pose2d, conf], axis=-1)
pose2d = torch.tensor(pose2d[:, :2], dtype=torch.float32).unsqueeze(0).to('cuda')  # (1, 28, 2)

# === Build Sparse Edge Index (matches training: 34 edges)
def build_sparse_edge_index():
    base_edges = [
        (0,1), (1,8), (8,12),
        (1,2), (2,3), (3,4),
        (1,5), (5,6), (6,7),
        (8,9), (9,10), (10,11),
        (8,13), (13,14), (14,15),
        (0,16), (0,17)
    ]
    # Add reverse edges to make the graph undirected
    bidirectional_edges = base_edges + [(j, i) for (i, j) in base_edges]
    edge_index = torch.tensor(bidirectional_edges, dtype=torch.long).t()  # Shape: [2, 34]
    return edge_index


edge_index = build_sparse_edge_index().to('cuda')

# === Run Inference ===
with torch.no_grad():   
    output_3d = model(pose2d, edge_index)  # Output: (1, 28, 3)
    output_3d = output_3d.squeeze(0).cpu().numpy()

# === Plot the 3D Pose
def plot_3d_pose(pose3d, edges, title="Predicted 3D Pose"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, j in edges:
        ax.plot([pose3d[i, 0], pose3d[j, 0]],
                [pose3d[i, 1], pose3d[j, 1]],
                [pose3d[i, 2], pose3d[j, 2]], 'bo-')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()

# === Define skeleton connections (same as edge_index)
edges = [
    (0,1), (1,8), (8,12),
    (1,2), (2,3), (3,4),
    (1,5), (5,6), (6,7),
    (8,9), (9,10), (10,11),
    (8,13), (13,14), (14,15),
    (0,16), (0,17)
]

plot_3d_pose(output_3d, edges)
