import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from model import PoseEstimator, device
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# --- Define skeletal connections ---
def create_edge_index():
    edges = [
        (11, 12), (11, 13), (13, 15),   # Left Arm
        (12, 14), (14, 16),             # Right Arm
        (11, 23), (12, 24),             # Torso
        (23, 25), (25, 27),             # Left Leg
        (24, 26), (26, 28),             # Right Leg
        (0, 15), (0, 16)                # Face
    ]
    edges = [(i, j) for i, j in edges if i < 28 and j < 28]
    edges += [(j, i) for i, j in edges]
    return edges

# --- Align using PCA ---
def align_pose_properly(pose3d):
    # Center at pelvis
    pose3d -= pose3d[1]

    # Flip X to match image orientation
    pose3d[:, 0] *= -1

    # Rotate around Z axis (camera orientation)
    theta = np.radians(30)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    pose3d = pose3d @ Rz.T

    # Scale based on shoulder width (11–12)
    shoulder_dist = np.linalg.norm(pose3d[11] - pose3d[12])
    if shoulder_dist > 1e-4:
        pose3d *= 180 / shoulder_dist

    return pose3d


# --- Equal scaling ---
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim3d([mid_x - max_range / 2, mid_x + max_range / 2])
    ax.set_ylim3d([mid_y - max_range / 2, mid_y + max_range / 2])
    ax.set_zlim3d([mid_z - max_range / 2, mid_z + max_range / 2])

# --- Color segment logic ---
def part_color(i, j):
    if i in [13, 15] or j in [13, 15]:
        return 'blue'      # Left Arm
    elif i in [14, 16] or j in [14, 16]:
        return 'green'     # Right Arm
    elif i in [25, 27] or j in [25, 27]:
        return 'orange'    # Left Leg
    elif i in [26, 28] or j in [26, 28]:
        return 'red'       # Right Leg
    elif i in [11, 12, 23, 24] or j in [11, 12, 23, 24]:
        return 'purple'    # Torso
    else:
        return 'gray'      # Face/Other

# --- Load model ---
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()
pose2d_stats = np.load("pose2d_mean_std.npy", allow_pickle=True)
pose2d_mean, pose2d_std = pose2d_stats[0][:2], pose2d_stats[1][:2]

# --- Load input image ---
image_path = "input.jpeg"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"❌ Image not found at {image_path}")

h, w = frame.shape[:2]
size = min(h, w)
frame = frame[(h - size)//2:(h + size)//2, (w - size)//2:(w + size)//2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --- Extract keypoints ---
pose = mp.solutions.pose.Pose(static_image_mode=True)
results = pose.process(rgb)
if not results.pose_landmarks:
    raise RuntimeError("❌ No person detected in the image.")

landmarks = results.pose_landmarks.landmark[:28]
keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
keypoints_2d = (keypoints_2d - pose2d_mean) / (pose2d_std + 1e-6)
input_tensor = torch.tensor(np.concatenate([keypoints_2d, np.zeros((28, 1))], axis=1),
                            dtype=torch.float32).unsqueeze(0).to(device)

# --- Predict 3D ---
edge_list = create_edge_index()
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
with torch.no_grad():
    pred_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

pred_3d = align_pose_properly(pred_3d)


# --- Plot 3D ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Accurate 3D Human Pose (PCA Aligned)")
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
ax.view_init(elev=30, azim=-60)
set_axes_equal(ax)
plt.tight_layout()
plt.savefig("final_accurate_3d_pose.png", dpi=300)
print("✅ Saved 3D pose visualization to final_accurate_3d_pose.png")
plt.show()
