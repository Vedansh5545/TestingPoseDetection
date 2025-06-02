import cv2
import numpy as np
import torch
from model import PoseEstimator, create_edge_index
import mediapipe as mp
import matplotlib.pyplot as plt

# === Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
edge_index = create_edge_index().to(device)

# === Load Model
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()

# === Read Image
image_path = "input.jpeg"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError("❌ Image not found.")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(frame_rgb)

if not results.pose_landmarks:
    print("⚠️ No landmarks found.")
    exit()

landmarks = results.pose_landmarks.landmark
if len(landmarks) < 28:
    print("⚠️ Not enough landmarks.")
    exit()

# === Extract full 3D joints from MediaPipe (x, y, z)
keypoints_3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])[:28]

# === Normalize using training stats
stats = np.load("pose2d_mean_std.npy")
mean, std = stats[0], stats[1]
keypoints_3d_norm = (keypoints_3d - mean) / (std + 1e-6)

input_tensor = torch.tensor(keypoints_3d_norm, dtype=torch.float32).unsqueeze(0).to(device)
edge_batched = edge_index.unsqueeze(0)

# === Inference
with torch.no_grad():
    out_3d = model(input_tensor, edge_batched).squeeze(0).cpu().numpy()

# === Projection to 2D for visualization
out_2d = out_3d[:, :2]
out_2d -= out_2d.min(axis=0)
out_2d /= out_2d.max()
out_2d *= 400

# === Draw
skeleton_edges = [
    (0, 1), (1, 8), (8, 12),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (8, 13), (13, 14), (14, 15),
    (0, 16), (0, 17)
]

plt.figure(figsize=(5, 5))
for i, j in skeleton_edges:
    if i < len(out_2d) and j < len(out_2d):
        x = [out_2d[i][0], out_2d[j][0]]
        y = [out_2d[i][1], out_2d[j][1]]
        plt.plot(x, y, 'g-', linewidth=2)

# Plot joints
for idx, (x, y) in enumerate(out_2d):
    plt.scatter(x, y, color='red')
    plt.text(x+2, y+2, str(idx), fontsize=7)

plt.title("Pose Graph (ViTPose-inspired GCN)")
plt.gca().invert_yaxis()
plt.axis('off')
plt.savefig("pose_graph_gcn_vitpose.png")
plt.show()
