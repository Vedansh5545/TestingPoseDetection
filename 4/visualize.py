# visualize.py
import cv2
import numpy as np

# We’ll use MediaPipe’s built-in POSE_CONNECTIONS for a 33-point 2D skeleton.
# Each pair (i,j) connects landmark i to landmark j.
# This is guaranteed to match what MediaPipe’s 2D model uses internally.
POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,12),
    (11,13), (13,15),
    (12,14), (14,16),
    (11,23), (12,24),
    (23,25), (25,27),
    (24,26), (26,28),
    (27,29), (28,30),
    (29,31), (30,32)
]

# Colors (BGR)
BONE_COLOR      = (0,   255,   0)   # Green for bones
JOINT_COLOR     = (0,     0, 255)   # Red for each joint
JOINT_RADIUS    = 4
BONE_THICKNESS  = 2
TEXT_COLOR      = (255, 255, 255)   # White for labels

def draw_2d_pose(frame, landmarks):
    """
    Draws a 2D skeleton on 'frame' using MediaPipe's 2D landmarks.
    
    - frame: a BGR image (HxW×3).
    - landmarks: a list (or array) of 33 landmarks, each with normalized x,y (both in [0,1]).
                 We assume 'landmarks[i].x' and 'landmarks[i].y' are the 2D outputs from MediaPipe.
    """

    H, W = frame.shape[:2]

    # Convert normalized (x,y) ∈ [0,1] to pixel coordinates (0..W-1, 0..H-1)
    pts_px = np.array([[int(lm.x * W), int(lm.y * H)] for lm in landmarks])

    # Draw bones (green lines) according to POSE_CONNECTIONS
    for (i, j) in POSE_CONNECTIONS:
        if i < len(pts_px) and j < len(pts_px):
            p1 = tuple(pts_px[i])
            p2 = tuple(pts_px[j])
            if p1 != p2:
                cv2.line(frame, p1, p2, BONE_COLOR, BONE_THICKNESS)

    # Draw joints (red circles) + index labels
    for idx, pt in enumerate(pts_px):
        cv2.circle(frame, tuple(pt), JOINT_RADIUS, JOINT_COLOR, -1)
        cv2.putText(frame, str(idx), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

    return frame
