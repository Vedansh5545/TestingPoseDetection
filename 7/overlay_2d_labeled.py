# overlay_2d_labeled.py

import argparse
import cv2
import numpy as np
import mediapipe as mp

from skeleton_utils import MPIINF_EDGES, MEDIAPIPE_TO_MPIINF

# ─── 1) Define 28 Joint Names ────────────────────────────────
JOINT_NAMES = [
    "pelvis",        # 0
    "right_hip",     # 1
    "right_knee",    # 2
    "right_ankle",   # 3
    "right_heel",    # 4
    "left_hip",      # 5
    "left_knee",     # 6
    "left_ankle",    # 7
    "left_heel",     # 8
    "spine_mid",     # 9
    "neck",          #10
    "head_top",      #11
    "left_shoulder", #12
    "left_elbow",    #13
    "left_wrist",    #14
    "right_shoulder",#15
    "right_elbow",   #16
    "right_wrist",   #17
    # remaining 10:
    "joint_18","joint_19","joint_20","joint_21",
    "joint_22","joint_23","joint_24","joint_25",
    "joint_26","joint_27"
]

# ─── 2) Major bones to label (i<j) ──────────────────────────
MAJOR_EDGES = {
    (0, 9):  "pelvis→spine_mid",
    (9,10):  "spine_mid→neck",
    (10,11): "neck→head_top",
    (11,15): "head_top→right_shoulder",
    (15,16): "right_shoulder→right_elbow",
    (16,17): "right_elbow→right_wrist",
    (11,12): "head_top→left_shoulder",
    (12,13): "left_shoulder→left_elbow",
    (13,14): "left_elbow→left_wrist",
    (0, 1):  "pelvis→right_hip",
    (1, 2):  "right_hip→right_knee",
    (2, 3):  "right_knee→right_ankle",
    (3, 4):  "right_ankle→right_heel",
    (0, 5):  "pelvis→left_hip",
    (5, 6):  "left_hip→left_knee",
    (6, 7):  "left_knee→left_ankle",
    (7, 8):  "left_ankle→left_heel",
}
BONE_LABELS = {**MAJOR_EDGES, **{(j,i):name for (i,j),name in MAJOR_EDGES.items()}}

# ─── 3) Drawing params ──────────────────────────────────────
BONE_COLOR     = (0,255,0)
JOINT_COLOR    = (0,0,255)
TEXT_COLOR     = (255,255,255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 2
FONT_SCALE     = 0.4
FONT_THICKNESS = 1
CONF_THRESH    = 0.1

def draw_2d_pose_with_labels(frame: np.ndarray, pose28: np.ndarray) -> np.ndarray:
    """
    Draws 28‐joint skeleton with semantic bone labels on major bones.
    """
    pts = pose28[:,:2].astype(int)
    # Bones + labels
    for i, j in MPIINF_EDGES:
        if pose28[i,2] > CONF_THRESH and pose28[j,2] > CONF_THRESH:
            p1, p2 = tuple(pts[i]), tuple(pts[j])
            cv2.line(frame, p1, p2, BONE_COLOR, BONE_THICKNESS)
            name = BONE_LABELS.get((i,j))
            if name:
                mx, my = (p1[0]+p2[0])//2 + 5, (p1[1]+p2[1])//2 - 5
                cv2.putText(frame, name.replace("_"," "), (mx,my),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    # Joints
    for idx, (x,y,conf) in enumerate(pose28):
        if conf > CONF_THRESH:
            cv2.circle(frame, (int(x),int(y)), JOINT_RADIUS, JOINT_COLOR, -1)
    return frame

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to input image")
    p.add_argument("-o","--output", default="overlay_labeled.jpg",
                   help="Where to save the labeled overlay")
    p.add_argument("--conf", type=float, default=CONF_THRESH,
                   help="Visibility threshold")
    args = p.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read '{args.image}'")
    H, W = frame.shape[:2]

    # 1) Detect 2D keypoints via MediaPipe
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        print("No pose detected.")
        return

    # 2) Build 28×3 pose array
    pose2d = np.zeros((28,3), dtype=np.float32)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None:
            continue
        lm = res.pose_landmarks.landmark[mp_i]
        pose2d[mpi_i] = [lm.x * W, (1 - lm.y) * H, lm.visibility]

    # 3) Draw & save
    vis = draw_2d_pose_with_labels(frame.copy(), pose2d)
    cv2.imwrite(args.output, vis)
    print("Saved labeled overlay to", args.output)

if __name__=="__main__":
    main()
