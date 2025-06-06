import cv2
import numpy as np

# === Skeleton Edges
SKELETON_EDGES = [
    (0, 1), (1, 8), (8, 12),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (8, 13), (13, 14), (14, 15),
    (0, 16), (0, 17)
]

# === Colors
EDGE_COLOR = (0, 255, 0)   # Green
JOINT_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White

IMG_WIDTH, IMG_HEIGHT = 640, 480

def draw_3d_pose(frame, joints_3d, image_dims=(IMG_WIDTH, IMG_HEIGHT), draw_head=True):
    w, h = image_dims

    if joints_3d.shape[1] < 2:
        raise ValueError("Expected 2D or 3D joint coordinates.")

    # === Clip 3D outliers
    joints_3d = np.clip(joints_3d, -2.5, 2.5)

    # === Project 3D to 2D
    joints_2d = joints_3d[:, :2] * np.array([w, h])
    joints_2d = np.nan_to_num(joints_2d, nan=0.0, posinf=0.0, neginf=0.0)
    joints_2d = np.clip(joints_2d, 0, min(w - 1, h - 1)).astype(int)

    # === Filter edges
    edges_to_draw = [e for e in SKELETON_EDGES if e[0] < len(joints_2d) and e[1] < len(joints_2d)]

    # === Draw bones (green lines)
    for i, j in edges_to_draw:
        pt1 = tuple(joints_2d[i])
        pt2 = tuple(joints_2d[j])
        if pt1 != pt2:  # avoid collapsed joints
            cv2.line(frame, pt1, pt2, EDGE_COLOR, 2)

    # === Draw joints (red dots) and optionally labels
    for idx, pt in enumerate(joints_2d):
        cv2.circle(frame, tuple(pt), 4, JOINT_COLOR, -1)
        cv2.putText(frame, str(idx), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

    return frame
