# visualize.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skeleton_utils import MPIINF_EDGES

# Visualization parameters
BONE_COLOR     = (0, 255, 0)
JOINT_COLOR    = (0, 0, 255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 2
CONF_THRESH    = 0.1

def draw_2d_pose_28(frame: np.ndarray, pose28: np.ndarray) -> np.ndarray:
    """
    Draws a 28-joint 2D pose on the provided BGR image frame.

    Args:
        frame: HxWx3 BGR image as a NumPy array
        pose28: (28,3) array of [x_px, y_px, confidence]
    Returns:
        Annotated frame with bones and joints drawn.
    """
    pts = pose28[:, :2].astype(int)

    # Draw bones
    for i, j in MPIINF_EDGES:
        if pose28[i, 2] > CONF_THRESH and pose28[j, 2] > CONF_THRESH:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), BONE_COLOR, BONE_THICKNESS)

    # Draw joints
    for idx, (x, y, conf) in enumerate(pose28):
        if conf > CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), JOINT_RADIUS, JOINT_COLOR, -1)
            cv2.putText(frame, str(idx), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

def plot_3d(pose3d: np.ndarray, elev: float = 20, azim: float = -70):
    # 1) Center on root (joint 0)
    root = pose3d[0:1, :]             
    pose3d_centered = pose3d - root    
    
    # 2) Flip X so it matches your 2D left-right
    pose3d_centered[:, 0] *= -1           
    # 3) Flip Z so “up” is positive
    pose3d_centered[:, 2] *= -1

    # 3) Now plot pose3d_centered instead of pose3d
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, j in MPIINF_EDGES:
        xs = [pose3d_centered[i,0], pose3d_centered[j,0]]
        ys = [pose3d_centered[i,1], pose3d_centered[j,1]]
        zs = [pose3d_centered[i,2], pose3d_centered[j,2]]
        ax.plot(xs, ys, zs, 'bo-', linewidth=2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1,1,1])  # ensure equal scaling on all axes
    plt.tight_layout()
    plt.show()
