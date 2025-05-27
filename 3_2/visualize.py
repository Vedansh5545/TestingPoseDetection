import cv2
import numpy as np

# === Define skeleton edges (MediaPipe 28 subset + optional head/face)
SKELETON_EDGES = [
    (11, 12),             # Shoulders
    (11, 13), (13, 15),   # Left Arm
    (12, 14), (14, 16),   # Right Arm
    (11, 23), (12, 24),   # Torso to hips
    (23, 25), (25, 27),   # Left Leg
    (24, 26), (26, 28),   # Right Leg
    # Optional head/face (safe)
    (0, 15), (0, 16), (15, 17), (16, 18)
]


# === Colors
EDGE_COLOR = (0, 255, 0)   # Green lines
JOINT_COLOR = (0, 0, 255)  # Red dots

# === Output image size (rescale projection)
IMG_WIDTH, IMG_HEIGHT = 640, 480


def draw_3d_pose(frame, joints_3d, image_dims=(IMG_WIDTH, IMG_HEIGHT), draw_head=True):
    """
    Projects 3D joint coordinates to 2D and draws them on the frame.
    :param frame: The input image
    :param joints_3d: (N, 3) array of 3D joint coordinates
    :param image_dims: Target image dimensions (W, H) for projection
    :param draw_head: Whether to draw optional head/face lines
    :return: Modified frame
    """
    w, h = image_dims

    if joints_3d.shape[1] < 2:
        raise ValueError("Expected 2D or 3D joint coordinates.")

    # === Project 3D to 2D using (x, y) scaled to image size
    joints_2d = joints_3d[:, :2] * np.array([w, h])
    joints_2d = np.nan_to_num(joints_2d, nan=0.0, posinf=0.0, neginf=0.0)
    joints_2d = np.clip(joints_2d, 0, min(frame.shape[1] - 1, frame.shape[0] - 1))
    joints_2d = joints_2d.astype(int)

    # === Select which edges to draw
    edges_to_draw = SKELETON_EDGES if draw_head else [e for e in SKELETON_EDGES if max(e) < 23]

    # === Draw bones
    for i, j in edges_to_draw:
        if i < len(joints_2d) and j < len(joints_2d):
            pt1 = tuple(joints_2d[i])
            pt2 = tuple(joints_2d[j])
            cv2.line(frame, pt1, pt2, EDGE_COLOR, 2)

    # === Draw joints
    for pt in joints_2d:
        cv2.circle(frame, tuple(pt), 4, JOINT_COLOR, -1)

    return frame
