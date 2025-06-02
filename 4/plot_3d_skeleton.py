import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
from matplotlib.lines import Line2D

# === 1) Detect 2D joints using MediaPipe (first 28 landmarks) ===
def extract_2d_keypoints(image_path):
    """
    Returns:
      - joints_2d: numpy array of shape (28, 2) in pixel coords
      - (w, h): image width and height
      - hips_y: average pixel-Y of the left & right hip
    """
    mp_pose = mp.solutions.pose
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            raise ValueError("No pose detected in the image.")

        raw_pts = []
        for lm in results.pose_landmarks.landmark:
            x_px = lm.x * w
            y_px = lm.y * h
            raw_pts.append([x_px, y_px])

        raw_pts = np.array(raw_pts)  # shape: (33, 2)

        # We only want the first 28 for consistency with MPI-INF indexing
        joints_2d = raw_pts[:28]

        # In MediaPipe, indices 23 & 24 are Left Hip and Right Hip
        left_hip_y  = raw_pts[23][1]
        right_hip_y = raw_pts[24][1]
        hips_y = (left_hip_y + right_hip_y) / 2.0

        return joints_2d, (w, h), hips_y

# === 2) Embed 2D joints into 3D (Z = 0), flipping Y so “up in image” = “up in plot” ===
def plot_flat_skeleton_with_labels_and_legend(joints_2d, hips_y, image_size):
    """
    joints_2d: (28,2) array of pixel coords
    hips_y: the pixel-Y coordinate of the hip line
    image_size: (w, h)
    """
    w, h = image_size

    # (A) Flip each Y so that smaller pixel-y → higher in plot
    pts3d = []
    for (x_px, y_px) in joints_2d:
        x_plot = x_px
        y_plot = h - y_px
        pts3d.append([x_plot, y_plot, 0.0])  # Z=0
    pts3d = np.array(pts3d)  # shape: (28, 3)

    # (B) Build edges from MediaPipe’s POSE_CONNECTIONS (filter < 28)
    mp_pose = mp.solutions.pose
    all_conns = mp_pose.POSE_CONNECTIONS  # set of (i,j) pairs for 33 landmarks

    edges = []
    for (i, j) in all_conns:
        if i < 28 and j < 28:
            edges.append((i, j))

    # Now classify each bone:
    # - If both joint-y_px < hips_y → “upper” (red)
    # - If both joint-y_px > hips_y → “lower” (blue)
    # - Else → “torso/mixed” (green)
    bone_colors = []
    bone_classes = []
    for (i, j) in edges:
        yi = joints_2d[i][1]
        yj = joints_2d[j][1]
        if yi < hips_y and yj < hips_y:
            bone_colors.append('r')  # upper body
            bone_classes.append('upper')
        elif yi > hips_y and yj > hips_y:
            bone_colors.append('b')  # lower body
            bone_classes.append('lower')
        else:
            bone_colors.append('g')  # crossing hip (torso/mixed)
            bone_classes.append('torso')

    # (C) Plotting
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("2D Pose in 3D Plane (Z=0) with Bone Labels")

    xs = pts3d[:, 0]
    ys = pts3d[:, 1]
    zs = pts3d[:, 2]  # all zeros

    # 1) Plot each joint as a red dot
    ax.scatter(xs, ys, zs, c='r', s=50)

    # 2) Draw each bone in its category color, and label it
    for idx_edge, (i, j) in enumerate(edges):
        x_line = [pts3d[i, 0], pts3d[j, 0]]
        y_line = [pts3d[i, 1], pts3d[j, 1]]
        z_line = [pts3d[i, 2], pts3d[j, 2]]
        color = bone_colors[idx_edge]
        cls   = bone_classes[idx_edge]
        ax.plot(x_line, y_line, z_line, c=color, linewidth=2)

        # Label the midpoint of this bone with its class
        mid_x = (x_line[0] + x_line[1]) / 2.0
        mid_y = (y_line[0] + y_line[1]) / 2.0
        mid_z = 0
        ax.text(mid_x, mid_y, mid_z, cls, color=color, fontsize=6)

    # 3) Optionally label each joint index for debugging
    for idx_pt, (x, y, z) in enumerate(pts3d):
        ax.text(x, y, z, str(idx_pt), color='black', fontsize=8)

    # 4) Label axes
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)  [flipped]")
    ax.set_zlabel("Z = 0")

    # 5) View straight down onto Z=0 plane
    ax.view_init(elev=90, azim=-90)

    # 6) Equalize X–Y scaling
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(0, 0)

    # 7) Add a legend explaining colors
    legend_elements = [
        Line2D([0], [0], color='r', lw=3, label='Upper-body bone'),
        Line2D([0], [0], color='b', lw=3, label='Lower-body bone'),
        Line2D([0], [0], color='g', lw=3, label='Torso/mixed bone')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


# === Main ===
if __name__ == "__main__":
    image_path = "input2.jpeg"  # ← Replace with your image filename
    joints_2d, (w, h), hips_y = extract_2d_keypoints(image_path)
    plot_flat_skeleton_with_labels_and_legend(joints_2d, hips_y, (w, h))
