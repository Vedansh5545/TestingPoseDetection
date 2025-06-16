# visualize.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skeleton_utils import COCO17_EDGES
from skeleton_utils import MPIINF_EDGES

def draw_coco17(frame, pose2d):
    for i,j in COCO17_EDGES:
        if pose2d[i,2]>0.1 and pose2d[j,2]>0.1:
            p1=(int(pose2d[i,0]),int(pose2d[i,1]))
            p2=(int(pose2d[j,0]),int(pose2d[j,1]))
            cv2.line(frame,p1,p2,(0,255,0),2)
    for idx,(x,y,c) in enumerate(pose2d):
        if c>0.1:
            cv2.circle(frame,(int(x),int(y)),4,(0,0,255),-1)
            cv2.putText(frame,str(idx),(int(x),int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return frame

def plot_3d(pose3d):
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111,projection='3d')
    for i,j in COCO17_EDGES:
        xs,ys,zs = pose3d[i,0],pose3d[i,1],pose3d[i,2],\
                  pose3d[j,0],pose3d[j,1],pose3d[j,2]
        ax.plot([pose3d[i,0],pose3d[j,0]],
                [pose3d[i,1],pose3d[j,1]],
                [pose3d[i,2],pose3d[j,2]],'bo-')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()



# Visualization parameters
BONE_COLOR     = (0, 255, 0)
JOINT_COLOR    = (0, 0, 255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 2
CONF_THRESH    = 0.1

def draw_2d_pose_28(frame: np.ndarray, pose28: np.ndarray) -> np.ndarray:
    """
    Draws a 28-joint 2D pose on the provided image frame.

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
