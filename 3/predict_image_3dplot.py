import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import PoseEstimator, create_edge_index, device

# === Load Model ===
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# === Load Normalization Stats ===
pose2d_stats = np.load("pose2d_mean_std.npy", allow_pickle=True)
pose2d_mean, pose2d_std = pose2d_stats[0][:2], pose2d_stats[1][:2]
print("✅ Loaded pose2d_mean_std.npy")

# === Load Image ===
image_path = "input.jpeg"
if not os.path.exists(image_path):
    raise FileNotFoundError("❌ input.jpeg not found.")
frame = cv2.imread(image_path)
h, w = frame.shape[:2]
size = min(h, w)
centered_frame = frame[(h - size) // 2:(h + size) // 2, (w - size) // 2:(w + size) // 2]
frame_rgb = cv2.cvtColor(centered_frame, cv2.COLOR_BGR2RGB)

# === Extract 2D Keypoints ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(frame_rgb)

if not results.pose_landmarks:
    print("🚫 No landmarks detected.")
    exit()

landmarks = results.pose_landmarks.landmark
if len(landmarks) < 28:
    print(f"⚠️ Only {len(landmarks)} landmarks detected. Expected 28.")
    exit()

keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks[:28]], dtype=np.float32)

# === Save Debug 2D Overlay ===
debug_frame = centered_frame.copy()
for (x, y) in (keypoints_2d * [w, h]).astype(int):
    cv2.circle(debug_frame, (x, y), 4, (0, 255, 0), -1)
cv2.imwrite("debug_input_2d.jpg", debug_frame)
print("✅ Saved 2D overlay to debug_input_2d.jpg")

# === Normalize and Prepare Input ===
keypoints_2d = (keypoints_2d - pose2d_mean) / (pose2d_std + 1e-6)
keypoints_3d_input = np.concatenate([keypoints_2d, np.zeros((28, 1), dtype=np.float32)], axis=-1)
input_tensor = torch.tensor(keypoints_3d_input, dtype=torch.float32).unsqueeze(0).to(device)

# === Run Model ===
edge_index = create_edge_index().to(device)
with torch.no_grad():
    out_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

# === Center around pelvis (joint 1) ===
out_3d -= out_3d[1]  # Make pelvis the origin

# === Collapse Check ===
if np.allclose(out_3d, out_3d[0], atol=1e-2):
    print("⚠️ Collapsed output detected.")
    exit()

# === Plot 3D Pose ===
def plot_3d_pose(joints_3d):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    ax.scatter(x, y, z, c='red', s=40, label='Joints')

    # Draw bones
    edges = create_edge_index().numpy().T
    for i, j in edges:
        if i < len(x) and j < len(x):
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue', linewidth=2)

    # Visual tweaks
    ax.set_title("Predicted 3D Pose")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-70)
    ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
    plt.legend()
    plt.tight_layout()
    plt.savefig("output_3d.png", dpi=300)
    print("✅ Saved 3D plot to output_3d.png")
    plt.show()

plot_3d_pose(out_3d)
