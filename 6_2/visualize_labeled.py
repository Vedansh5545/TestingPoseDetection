# visualize_labeled.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skeleton_utils import MPIINF_EDGES
from bone_labels import BONE_LABELS_SYM

# drawing params
BONE_COLOR     = (0,255,0)
JOINT_COLOR    = (0,0,255)
TEXT_COLOR     = (255,255,255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 2
FONT_SCALE     = 0.3
FONT_THICKNESS = 1
CONF_THRESH    = 0.1

def draw_2d_pose_with_labels(frame: np.ndarray, pose28: np.ndarray) -> np.ndarray:
    """
    Draws 28‐joint skeleton with bone‐name labels at each midpoint.
    """
    pts = pose28[:,:2].astype(int)
    # Draw bones and annotate names
    for i,j in MPIINF_EDGES:
        if pose28[i,2] > CONF_THRESH and pose28[j,2] > CONF_THRESH:
            p1, p2 = tuple(pts[i]), tuple(pts[j])
            cv2.line(frame, p1, p2, BONE_COLOR, BONE_THICKNESS)
            # midpoint
            mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
            name = BONE_LABELS_SYM.get((i,j))
            if name:
                cv2.putText(frame, name.replace("_"," "), (mx,my),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    # Draw joints
    for idx, (x,y,conf) in enumerate(pose28):
        if conf > CONF_THRESH:
            cv2.circle(frame, (int(x),int(y)), JOINT_RADIUS, JOINT_COLOR, -1)
            cv2.putText(frame, str(idx), (int(x),int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return frame

def plot_3d(pose3d: np.ndarray, elev: float = 20, azim: float = -70):
    """
    3D plot (unchanged).
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    for i,j in MPIINF_EDGES:
        xs = [pose3d[i,0], pose3d[j,0]]
        ys = [pose3d[i,1], pose3d[j,1]]
        zs = [pose3d[i,2], pose3d[j,2]]
        ax.plot(xs, ys, zs, 'bo-', linewidth=2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()
