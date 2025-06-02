import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import PoseEstimator, device


def create_edge_index():
    edges = [
        (0, 11), (11, 12), (11, 13), (13, 15),  # Spine + Left Arm
        (12, 14), (14, 16),                    # Right Arm
        (11, 23), (12, 24),                    # Spine to Hips
        (23, 25), (25, 27),                    # Left Leg
        (24, 26), (26, 28)                     # Right Leg
    ]
    edges = [(i, j) for i, j in edges if i < 28 and j < 28]
    edges += [(j, i) for i, j in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def normalize_bone_lengths(joints_3d, skeleton, target_length=100):
    for i, j in skeleton:
        bone = joints_3d[j] - joints_3d[i]
        length = np.linalg.norm(bone)
        if length > 1e-4:
            joints_3d[j] = joints_3d[i] + (bone / length) * target_length
    return joints_3d


def align_pose_to_image(out_3d):
    # Flip X-axis to match image orientation
    out_3d[:, 0] *= -1

    # Rotate around Z-axis (camera facing)
    theta_deg = 30
    theta = np.radians(theta_deg)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    out_3d = out_3d @ R_z.T

    # Scale based on shoulder width (between joints 11 and 12)
    shoulder_width = np.linalg.norm(out_3d[11] - out_3d[12])
    if shoulder_width > 1e-4:
        scale_factor = 180 / shoulder_width
        out_3d *= scale_factor

    return out_3d


# === Load model and stats
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()
pose2d_stats = np.load("pose2d_mean_std.npy", allow_pickle=True)
pose2d_mean, pose2d_std = pose2d_stats[0][:2], pose2d_stats[1][:2]

# === Load and crop image
image_path = "input.jpeg"
frame = cv2.imread(image_path)
h, w = frame.shape[:2]
size = min(h, w)
frame = frame[(h - size)//2:(h + size)//2, (w - size)//2:(w + size)//2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === Extract 2D keypoints
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(rgb)

if not results.pose_landmarks:
    print("❌ No landmarks detected.")
    exit()

landmarks = results.pose_landmarks.landmark
if len(landmarks) < 28:
    print(f"⚠️ Only {len(landmarks)} landmarks detected.")
    exit()

keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks[:28]], dtype=np.float32)
debug = frame.copy()
for (x, y) in (keypoints_2d * [w, h]).astype(int):
    cv2.circle(debug, (x, y), 4, (0, 255, 0), -1)
cv2.imwrite("debug_input_2d.jpg", debug)

# === Normalize and prepare input
keypoints_2d = (keypoints_2d - pose2d_mean) / (pose2d_std + 1e-6)
keypoints_3d_input = np.concatenate([keypoints_2d, np.zeros((28, 1), dtype=np.float32)], axis=-1)
input_tensor = torch.tensor(keypoints_3d_input, dtype=torch.float32).unsqueeze(0).to(device)

edge_index = create_edge_index().to(device)
with torch.no_grad():
    out_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

# === Align pose to match real image appearance
out_3d -= out_3d[1]
out_3d = align_pose_to_image(out_3d)

# === Collapse check
if np.allclose(out_3d, out_3d[0], atol=1e-2):
    print("⚠️ Collapsed output.")
    exit()

# === Extract skeleton and joints
skeleton = create_edge_index().numpy().T
used = np.unique(skeleton.flatten())
x, y, z = out_3d[used, 0], out_3d[used, 1], out_3d[used, 2]

# === Save pose to CSV
np.savetxt("predicted_3d_pose.csv", out_3d, delimiter=",", header="x,y,z", comments='')
print("✅ Saved 3D coordinates to predicted_3d_pose.csv")

# === Limb coloring
upper_body_joints = {11, 12, 13, 14, 15, 16}
lower_body_joints = {23, 24, 25, 26, 27, 28}

# === Plot the 3D pose
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='red', s=40, label='Joints')

plotted_colors = set()
for i, j in skeleton:
    if i in upper_body_joints or j in upper_body_joints:
        color = 'blue'
        label = 'Upper Body'
    elif i in lower_body_joints or j in lower_body_joints:
        color = 'green'
        label = 'Lower Body'
    else:
        color = 'purple'
        label = 'Spine/Core'

    ax.plot(
        [out_3d[i, 0], out_3d[j, 0]],
        [out_3d[i, 1], out_3d[j, 1]],
        [out_3d[i, 2], out_3d[j, 2]],
        color=color,
        linewidth=2,
        label=label if color not in plotted_colors else None
    )
    plotted_colors.add(color)

ax.set_title("Corrected 3D Pose")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=-70)
ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
plt.legend()
plt.tight_layout()
plt.savefig("output_3d_v2.png", dpi=300)
print("✅ Saved 3D plot to output_3d_v2.png")
plt.show()
