# visualize.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skeleton_utils import COCO17_EDGES

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
