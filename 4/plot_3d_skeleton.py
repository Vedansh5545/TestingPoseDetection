import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp

# === 1) Detect 2D joints using MediaPipe ===
def extract_2d_keypoints(image_path):
    mp_pose = mp.solutions.pose

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            raise ValueError("No pose detected.")

        keypoints = []
        h, w = image.shape[:2]
        for lm in results.pose_landmarks.landmark:
            # Convert normalized [0,1] → pixel coordinates
            keypoints.append([lm.x * w, lm.y * h])

        keypoints = np.array(keypoints)  # shape: (33, 2)
        return keypoints[:28]           # keep first 28 joints

# === 2) Plot 2D joints in a 3D plane (Z=0) ===
def plot_flat_skeleton_3d(joints_2d):
    """
    joints_2d: numpy array of shape (28,2), pixel coords.
    We embed them into 3D as (x, y, 0).
    """
    # Create (28,3) by appending a zero Z-coordinate
    joints_3d = np.concatenate([joints_2d, np.zeros((28,1))], axis=1)

    # Define the same skeleton connectivity used in MPI-INF/GCN
    edges = [
        (0, 1), (1, 8), (8, 12),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11),
        (8, 13), (13, 14), (14, 15),
        (0, 16), (0, 17)
    ]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("2D Pose Plotted in 3D Plane (Z=0)")

    # Plot each joint as a red dot
    xs = joints_3d[:, 0]
    ys = joints_3d[:, 1]
    zs = joints_3d[:, 2]  # all zeros
    ax.scatter(xs, ys, zs, c='r', s=50)

    # Draw bones (green lines) connecting the joints
    for i, j in edges:
        x_line = [joints_3d[i, 0], joints_3d[j, 0]]
        y_line = [joints_3d[i, 1], joints_3d[j, 1]]
        z_line = [joints_3d[i, 2], joints_3d[j, 2]]
        ax.plot(x_line, y_line, z_line, c='g', linewidth=2)

    # Optionally label each joint index for debugging
    for idx_pt, (x, y, z) in enumerate(joints_3d):
        ax.text(x, y, z, str(idx_pt), color='blue', fontsize=8)

    # Label axes
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_zlabel("Z = 0")

    # Because all points lie on Z=0, adjust the view angle
    ax.view_init(elev=90, azim=-90)  # top-down view so you see X vs. Y plane
    # Alternatively, use a slight tilt to see the flatness:
    # ax.view_init(elev=60, azim=-90)

    # Equalize X/Y axis scaling so the pose isn't distorted
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(0, 0)  # all points on zero plane

    plt.tight_layout()
    plt.show()


# === Main Execution ===
if __name__ == "__main__":
    image_path = "input.jpeg"  # Replace with your image filename

    # 1) Extract 2D joints
    joints_2d = extract_2d_keypoints(image_path)  # shape: (28,2)

    # 2) Plot them flat in 3D
    plot_flat_skeleton_3d(joints_2d)
