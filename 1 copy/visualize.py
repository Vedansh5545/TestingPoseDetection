# visualize.py
import cv2
import numpy as np

# Define skeleton connections (simple chain)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (0, 16), (15, 17), (16, 18)
]

# Colors
EDGE_COLOR = (0, 255, 0)
JOINT_COLOR = (0, 0, 255)

# Image dimensions
IMG_WIDTH, IMG_HEIGHT = 640, 480


def draw_3d_pose(frame, joints_3d):
    """
    Projects 3D joints onto the 2D image and draws connections
    """
    joints_2d = joints_3d[:, :2] * np.array([IMG_WIDTH, IMG_HEIGHT])  # Normalize to image size
    joints_2d = joints_2d.astype(int)
    joints_2d = np.nan_to_num(joints_2d, nan=0.0, posinf=0.0, neginf=0.0)
    joints_2d = np.clip(joints_2d, 0, min(frame.shape[1]-1, frame.shape[0]-1))
    joints_2d = joints_2d.astype(int)


    for i, j in SKELETON_EDGES:
        if i < len(joints_2d) and j < len(joints_2d):
            pt1 = tuple(joints_2d[i])
            pt2 = tuple(joints_2d[j])
            cv2.line(frame, pt1, pt2, EDGE_COLOR, 2)

    for pt in joints_2d:
        cv2.circle(frame, tuple(pt), 4, JOINT_COLOR, -1)

    return frame
