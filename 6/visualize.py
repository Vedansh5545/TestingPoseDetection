# visualize_28.py

import cv2
import numpy as np
from skeleton_utils import MPIINF_EDGES

BONE_COLOR     = (0,255,0)
JOINT_COLOR    = (0,0,255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 2
CONF_THRESH    = 0.1

def draw_2d_pose_28(frame: np.ndarray, pose28: np.ndarray) -> np.ndarray:
    """
    Draws the 28-joint 2D pose on `frame`.
    pose28: np.array shape (28,3): [x_px, y_px, conf].
    """
    pts = pose28[:,:2].astype(int)
    # draw bones
    for i,j in MPIINF_EDGES:
        if pose28[i,2] > CONF_THRESH and pose28[j,2] > CONF_THRESH:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), BONE_COLOR, BONE_THICKNESS)
    # draw joints
    for idx,(x,y,conf) in enumerate(pose28):
        if conf > CONF_THRESH:
            cv2.circle(frame, (int(x),int(y)), JOINT_RADIUS, JOINT_COLOR, -1)
            cv2.putText(frame, str(idx), (int(x),int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return frame
