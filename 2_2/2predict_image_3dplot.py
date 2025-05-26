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
model_path = "model_weights.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model weights not found at: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Load Normalization Stats ===
pose2d_mean, pose2d_std = np.load("pose2d_mean_std.npy", allow_pickle=True)
print("✅ Loaded pose2d_mean_std.npy")

# === Load Image ===
image_path = "input.jpeg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Input image not found at: {image_path}")
frame = cv2.imread(image_path)
h, w = frame.shape[:2]
size = min(h, w)
centered_frame = frame[(h - size) // 2:(h + size) // 2, (w - size) // 2:(w + size) // 2]
frame_rgb = cv2.cvtColor(centered_frame, cv2.COLOR_BGR2RGB)

# === Run MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(frame_rgb)

if not results.pose_landmarks:
    print("🚫 No person detected.")
    exit()

landmarks = results.pose_landmarks.landmark
if len(landmarks) < 28:
    print(f"⚠️ Only {len(landmarks)} landmarks detected. Expected at least 28.")
    exit()

# === Extract and Normalize Keypoints ===
keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks[:28]], dtype=np.float32)
print("📌 Raw keypoints:", keypoints_2d)

# Save input overlay
debug_frame = centered_frame.copy()
for (x, y) in (keypoints_2d * [w, h]).astype(int):
    cv2.circle(debug_frame, (x, y), 4, (0, 255, 0), -1)
cv2.imwrite("debug_input_2d.jpg", debug_frame)
print("✅ Saved input overlay to debug_input_2d.jpg")

# Normalize and pad to [28, 3]
keypoints_2d = (keypoints_2d - pose2d_mean) / (pose2d_std + 1e-6)
keypoints_3d_input = np.concatenate([keypoints_2d, np.zeros((28, 1), dtype=np.float32)], axis=-1)

input_tensor = torch.tensor(keypoints_3d_input, dtype=torch.float32).unsqueeze(0).to(device)
print("📥 Input tensor shape:", input_tensor.shape)
print("📥 Sample input (first 3):", input_tensor[0, :3])

# === Prepare edge index ===
edge_index = create_edge_index().to(device)

# === Inference ===
with torch.no_grad():
    out_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

print("📤 Output shape:", out_3d.shape)
print("📤 First 3 predicted keypoints:\n", out_3d[:3])

if np.allclose(out_3d, out_3d[0]):
    print("⚠️ Warning: All 3D outputs are nearly identical. Model may be collapsed.")
    exit()

# === Plotting Function ===
def plot_3d_pose(joints_3d):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    ax.scatter(x, y, z, c='red', s=30)

    skeleton = [
        (0,1), (1,2), (2,3), (3,4),       # right arm
        (1,5), (5,6), (6,7),              # left arm
        (1,8), (8,9), (9,10), (10,11),    # right leg
        (8,12), (12,13), (13,14)          # left leg
    ]
    for i, j in skeleton:
        if i < len(x) and j < len(x):
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue', linewidth=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-70)
    plt.title("3D Pose Visualization")
    plt.tight_layout()
    plt.savefig("output_3d.png")
    print("✅ Saved 3D plot to output_3d.png")
    plt.show()

# === Visualize Prediction ===
plot_3d_pose(out_3d)
