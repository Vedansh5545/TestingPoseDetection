import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PoseEstimator as Pose3DModel
from torch_geometric.utils import dense_to_sparse

# === Force CPU for inference to avoid CUDA OOM
device = torch.device("cpu")

# === Load model
model = Pose3DModel().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()

# === Define bone labels (MediaPipe style indices)
bone_names = {
    (11, 13): "Left Upper Arm",
    (13, 15): "Left Lower Arm",
    (12, 14): "Right Upper Arm",
    (14, 16): "Right Lower Arm",
    (23, 25): "Left Thigh",
    (25, 27): "Left Calf",
    (24, 26): "Right Thigh",
    (26, 28): "Right Calf",
    (11, 12): "Shoulders",
    (23, 24): "Hips",
    (11, 23): "Left Torso",
    (12, 24): "Right Torso"
}

# === Biological proportions (approximate)
bone_proportions = {
    "Upper Arm": 0.186,
    "Lower Arm": 0.146,
    "Thigh": 0.245,
    "Calf": 0.246,
    "Torso": 0.3
}

# === Load 2D input pose
data = np.load("mpi_inf_combined.npz")
print(data.files)
pose2d = torch.tensor(data['pose2d'], dtype=torch.float32).to(device)

# === Visibility mask
visibility = torch.isnan(pose2d).any(dim=-1)
pose2d[visibility] = 0  # fill missing with zeros

# === Define graph structure
num_joints = 33
adj_matrix = np.ones((num_joints, num_joints)) - np.eye(num_joints)
edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix))
edge_index = edge_index.to(device)

# === Hallucinate missing joints using symmetry
def hallucinate(pose, mask):
    pose = pose.clone()
    mirror_map = {
        11: 12, 13: 14, 15: 16, 23: 24, 25: 26, 27: 28,
        12: 11, 14: 13, 16: 15, 24: 23, 26: 25, 28: 27
    }
    for i in range(pose.shape[1]):
        if mask[0, i]:
            if i in mirror_map and not mask[0, mirror_map[i]]:
                pose[0, mirror_map[i]] = pose[0, i] * torch.tensor([-1.0, 1.0], dtype=torch.float32).to(device)
    return pose

pose2d_filled = hallucinate(pose2d, visibility)

# === Inference
with torch.no_grad():
    input_with_conf = torch.cat([
        pose2d_filled, torch.ones_like(pose2d_filled[..., :1])
    ], dim=-1).to(device)  # Shape: (1, 33, 3)

    output_3d = model(input_with_conf, edge_index)
    output_3d = output_3d.squeeze(0).cpu().numpy()

# === Plotting
def plot_3d(pose3d, bones, title="3D Pose"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, j in bones:
        ax.plot([pose3d[i, 0], pose3d[j, 0]],
                [pose3d[i, 1], pose3d[j, 1]],
                [pose3d[i, 2], pose3d[j, 2]], 'bo-')
    ax.set_title(title)
    ax.view_init(elev=20, azim=-70)
    plt.tight_layout()
    plt.show()

edges = list(bone_names.keys())
plot_3d(output_3d, edges, title="Bone-Aware 3D Pose Output")
