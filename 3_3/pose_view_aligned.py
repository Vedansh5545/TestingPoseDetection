import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from model import PoseEstimator, device
from mpl_toolkits.mplot3d import Axes3D

# ----------------------
# Define Skeleton
# ----------------------
def create_edge_index():
    edges = [
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28-1),
        (0, 15), (0, 16)
    ]
    edges += [(j, i) for i, j in edges]
    return edges

# ----------------------
# Proper anatomical alignment (no PCA)
# ----------------------
def align_pose_properly(pose3d):
    pose3d -= pose3d[1]  # center at pelvis
    pose3d[:, 0] *= -1   # flip X to match image

    # Rotate 30 degrees around Z axis
    theta = np.radians(30)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    pose3d = pose3d @ Rz.T

    # Scale based on shoulder width
    scale = np.linalg.norm(pose3d[11] - pose3d[12])
    if scale > 1e-5:
        pose3d *= 180 / scale
    return pose3d

# ----------------------
# Segment Color
# ----------------------
def part_color(i, j):
    if i in [13, 15] or j in [13, 15]: return 'blue'
    elif i in [14, 16] or j in [14, 16]: return 'green'
    elif i in [25, 27] or j in [25, 27]: return 'orange'
    elif i in [26, 28] or j in [26, 28]: return 'red'
    elif i in [11, 12, 23, 24] or j in [11, 12, 23, 24]: return 'purple'
    else: return 'gray'

# ----------------------
# Set Axes Equal
# ----------------------
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    max_range = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0])
    ) / 2.0
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# ----------------------
# Load and Predict
# ----------------------
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()
pose2d_stats = np.load("pose2d_mean_std.npy", allow_pickle=True)
pose2d_mean, pose2d_std = pose2d_stats[0][:2], pose2d_stats[1][:2]

frame = cv2.imread("input.jpeg")
if frame is None:
    raise FileNotFoundError("❌ Image not found.")
h, w = frame.shape[:2]
size = min(h, w)
frame = frame[(h - size)//2:(h + size)//2, (w - size)//2:(w + size)//2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

pose = mp.solutions.pose.Pose(static_image_mode=True)
results = pose.process(rgb)
if not results.pose_landmarks:
    raise RuntimeError("❌ No person detected.")

landmarks = results.pose_landmarks.landmark[:28]
keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
keypoints_2d = (keypoints_2d - pose2d_mean) / (pose2d_std + 1e-6)
input_tensor = torch.tensor(
    np.concatenate([keypoints_2d, np.zeros((28, 1))], axis=1),
    dtype=torch.float32).unsqueeze(0).to(device)

edge_list = create_edge_index()
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
with torch.no_grad():
    pred_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()
pred_3d = align_pose_properly(pred_3d)

# ----------------------
# Plot Aligned 3D Pose
# ----------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Human Pose (Image View Aligned)")

ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], color='black', s=30)
for i, j in edge_list:
    if i < 28 and j < 28:
        ax.plot(
            [pred_3d[i, 0], pred_3d[j, 0]],
            [pred_3d[i, 1], pred_3d[j, 1]],
            [pred_3d[i, 2], pred_3d[j, 2]],
            color=part_color(i, j), linewidth=3
        )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=15, azim=-85)  # View matching photo angle
set_axes_equal(ax)
plt.tight_layout()
plt.savefig("view_aligned_3d_pose.png", dpi=300)
print("✅ Saved aligned 3D pose to view_aligned_3d_pose.png")
plt.show()
