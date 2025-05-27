import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import PoseEstimator, create_edge_index, device

model = PoseEstimator().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

pose2d_mean, pose2d_std = np.load("pose2d_mean_std.npy")

image_path = "input.jpeg"
frame = cv2.imread(image_path)
h, w = frame.shape[:2]
size = min(h, w)
centered_frame = frame[(h - size) // 2:(h + size) // 2, (w - size) // 2:(w + size) // 2]
frame_rgb = cv2.cvtColor(centered_frame, cv2.COLOR_BGR2RGB)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(frame_rgb)

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    if len(landmarks) < 28:
        print("⚠️ Not enough keypoints detected.")
        exit()

    keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks])[:28]
    keypoints_2d = (keypoints_2d - pose2d_mean) / (pose2d_std + 1e-6)
    input_tensor = torch.tensor(keypoints_2d, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        edge_index = create_edge_index()
        out_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

    def plot_3d_pose(joints_3d):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
        ax.scatter(x, y, z, c='red', s=20)

        for i, j in [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),
                     (8,9),(9,10),(10,11),(8,12),(12,13),(13,14)]:
            if i < len(x) and j < len(x):
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1,1,1])  # Equal aspect ratio
        plt.title("3D Pose Visualization")
        plt.tight_layout()
        plt.savefig("output_3d.png")
        print("✅ Saved 3D plot to output_3d.png")
        plt.show()

    plot_3d_pose(out_3d)
else:
    print("No person detected in the image.")
