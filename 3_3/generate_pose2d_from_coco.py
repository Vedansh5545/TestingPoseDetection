import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# Config
image_dir = "E:/train2017/train2017"  # Path to COCO images
output_file = "pose2d_mediapipe_28.npz"
norm_stats_file = "pose2d_mean_std.npy"
max_images = 10000

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
pose2d_all = []

print("🔄 Generating pose2d data using MediaPipe...")

for idx, filename in tqdm(enumerate(os.listdir(image_dir))):
    if not filename.lower().endswith(".jpg"):
        continue
    if idx >= max_images:
        break

    path = os.path.join(image_dir, filename)
    image = cv2.imread(path)
    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        continue

    landmarks = results.pose_landmarks.landmark
    if len(landmarks) < 28:
        continue

    keypoints = np.array([[landmarks[i].x, landmarks[i].y] for i in range(28)], dtype=np.float32)
    keypoints = np.clip(keypoints, 0, 1)
    pose2d_all.append(keypoints)

# Convert to NumPy array
pose2d_all = np.array(pose2d_all)
np.savez(output_file, pose2d=pose2d_all)

# Compute and save mean/std
pose2d_mean = pose2d_all.mean(axis=(0, 1))  # shape (2,)
pose2d_std = pose2d_all.std(axis=(0, 1))    # shape (2,)
np.save(norm_stats_file, np.array([pose2d_mean, pose2d_std]))

print(f"✅ Saved {len(pose2d_all)} pose samples to {output_file}")
print(f"✅ Saved mean/std to {norm_stats_file}")
print(f"ℹ️ Mean: {pose2d_mean}, Std: {pose2d_std}")
