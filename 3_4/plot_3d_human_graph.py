import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from model import PoseEstimator, device, create_edge_index
from mpl_toolkits.mplot3d import Axes3D

# === 1. Setup
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()
pose2d_stats = np.load("pose2d_mean_std.npy", allow_pickle=True)
pose2d_mean, pose2d_std = pose2d_stats[0][:2], pose2d_stats[1][:2]

# === 2. Load Image
img_path = "input.jpeg"
frame = cv2.imread(img_path)
h, w = frame.shape[:2]
size = min(h, w)
frame = frame[(h-size)//2:(h+size)//2, (w-size)//2:(w+size)//2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === 3. 2D Keypoint Detection
pose = mp.solutions.pose.Pose(static_image_mode=True)
results = pose.process(rgb)
if not results.pose_landmarks:
    raise RuntimeError("No landmarks detected")

keypoints_2d = np.array([[lm.x * size, lm.y * size] for lm in results.pose_landmarks.landmark[:28]], dtype=np.float32)
normalized_2d = (keypoints_2d / size - pose2d_mean) / (pose2d_std + 1e-6)
input_tensor = torch.tensor(np.concatenate([normalized_2d, np.zeros((28,1))], axis=1), dtype=torch.float32).unsqueeze(0).to(device)

# === 4. Predict 3D Pose
edge_index = create_edge_index().to(device)
with torch.no_grad():
    joints_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

# === 5. Normalize + Realistic Bone Lengths
joints_3d -= joints_3d[1]  # Center at pelvis
joints_3d[:, 0] *= -1       # Flip X for visual match

# === 6. Define Skeleton + Colors
edges = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 27)
]
def color(i, j):
    if {i, j} & {13, 15, 11}: return 'blue'
    if {i, j} & {14, 16, 12}: return 'green'
    if {i, j} & {23, 25, 27}: return 'orange'
    if {i, j} & {24, 26, 28}: return 'red'
    return 'purple'

# === 7. Plot 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Human Pose (Real-Scale Aligned)")

for i, j in edges:
    if i < 28 and j < 28:
        ax.plot([joints_3d[i,0], joints_3d[j,0]],
                [joints_3d[i,1], joints_3d[j,1]],
                [joints_3d[i,2], joints_3d[j,2]],
                color=color(i,j), linewidth=3)

ax.scatter(joints_3d[:,0], joints_3d[:,1], joints_3d[:,2], color='black', s=30)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=-75)
ax.set_box_aspect([1,1,1])
plt.tight_layout()
plt.savefig("accurate_pose_3d_graph.png", dpi=300)
plt.show()
print("✅ Saved accurate 3D graph as accurate_pose_3d_graph.png")
