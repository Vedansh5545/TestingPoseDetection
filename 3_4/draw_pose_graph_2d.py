import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import PoseEstimator, create_edge_index

# Load and preprocess input
image_path = "input.jpeg"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError("Image not found.")

# === Setup MediaPipe
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(frame_rgb)

if not results.pose_landmarks:
    print("No person detected.")
    exit()

landmarks = results.pose_landmarks.landmark
if len(landmarks) < 28:
    print("Not enough landmarks detected.")
    exit()

keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks])[:28]
mean = keypoints_2d.mean(axis=0)
std = keypoints_2d.std(axis=0)
normed_kps = (keypoints_2d - mean) / (std + 1e-6)
input_tensor = torch.tensor(
    np.concatenate([normed_kps, np.zeros((28, 1))], axis=1),
    dtype=torch.float32
).unsqueeze(0)

# === Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()
edge_index = create_edge_index().to(device)

with torch.no_grad():
    out_3d = model(input_tensor.to(device), edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

# === Align + project 3D to 2D
out_3d[:, 0] *= -1  # Flip X
theta = np.radians(30)
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1]
])
out_3d = out_3d @ R_z.T
if 11 < len(out_3d) and 12 < len(out_3d):
    shoulder_width = np.linalg.norm(out_3d[11] - out_3d[12])
    if shoulder_width > 1e-4:
        scale = 180 / shoulder_width
        out_3d *= scale

joints_2d = out_3d[:, :2]
joints_2d -= joints_2d.min(axis=0)  # shift to (0, 0)
joints_2d /= joints_2d.max()        # scale to (0,1)
joints_2d *= 400                    # fit in a 400x400 canvas

# === Draw using matplotlib
edges = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10)
]
# Note: Clip to max 28
edges = [(i, j) for i, j in edges if i < 28 and j < 28]

plt.figure(figsize=(5, 5))
for i, j in edges:
    if i < len(joints_2d) and j < len(joints_2d):
        x = [joints_2d[i][0], joints_2d[j][0]]
        y = [joints_2d[i][1], joints_2d[j][1]]
        plt.plot(x, y, color='green', linewidth=2)

# Plot joints
for idx, (x, y) in enumerate(joints_2d):
    plt.scatter(x, y, c='red')
    plt.text(x+3, y+3, str(idx), fontsize=8, color='black')

plt.title("2D Pose Graph from Predicted Joints")
plt.axis('off')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("pose_graph_2d.png")
plt.show()

print("Pose graph saved as 'pose_graph_2d.png'.")