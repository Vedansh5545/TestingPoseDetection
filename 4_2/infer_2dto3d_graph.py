import cv2
import numpy as np
import mediapipe as mp
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === STEP 1: Extract 2D Keypoints ===
def extract_keypoints_from_image(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if not result.pose_landmarks:
        raise ValueError("No pose detected.")

    landmarks = result.pose_landmarks.landmark
    joints_2d = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in landmarks])
    return joints_2d[:28]  # Get first 28 joints (custom subset)

# === STEP 2: Define bone structure and lengths ===
edges = [
    (0,1), (1,2), (2,3), (3,4),       # Left arm
    (1,5), (5,6), (6,7),             # Right arm
    (1,8), (8,9),                    # Spine
    (9,10), (10,11),                 # Left leg
    (9,12), (12,13)                  # Right leg
]

bone_lengths_cm = {
    (0,1): 25, (1,2): 18, (2,3): 30, (3,4): 25,
    (1,5): 18, (5,6): 30, (6,7): 25,
    (1,8): 30, (8,9): 25,
    (9,10): 40, (10,11): 40,
    (9,12): 40, (12,13): 40
}

# === STEP 3: Lift to 3D using bone lengths ===
def lift_to_3d(joints_2d, edges, bone_lengths):
    joints_3d = np.hstack([joints_2d, np.zeros((joints_2d.shape[0], 1))])
    joints_3d[1, 2] = 0  # anchor neck z = 0

    for (i, j) in edges:
        x1, y1 = joints_3d[i, :2]
        x2, y2 = joints_3d[j, :2]
        d2d = np.linalg.norm([x2 - x1, y2 - y1])
        L = bone_lengths.get((i, j)) or bone_lengths.get((j, i))
        if L is None or d2d == 0:
            continue
        dz = np.sqrt(max(L**2 - d2d**2, 0))
        joints_3d[j, 2] = joints_3d[i, 2] + dz
    return joints_3d

# === STEP 4: Plot in 3D ===
def plot_3d_pose(pose3d, edges):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, j in edges:
        ax.plot([pose3d[i, 0], pose3d[j, 0]],
                [pose3d[i, 1], pose3d[j, 1]],
                [pose3d[i, 2], pose3d[j, 2]], 'ro-')
    ax.set_title("3D Human Pose (Lifted from 2D + Bio-Lengths)")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_zlabel("Z (depth, cm)")
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()

# === Run everything ===
if __name__ == "__main__":
    image_path = "output.jpg"  # or input.jpeg if using the original image
    joints_2d = extract_keypoints_from_image(image_path)
    joints_3d = lift_to_3d(joints_2d, edges, bone_lengths_cm)
    plot_3d_pose(joints_3d, edges)
