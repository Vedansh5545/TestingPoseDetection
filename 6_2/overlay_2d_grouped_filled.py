# overlay_2d_grouped_filled.py

import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark

from skeleton_utils import MEDIAPIPE_TO_MPIINF    # maps MP idx → MPIINF idx
from skeleton_utils import MPIINF_EDGES           # your 28-joint edges
from missing_bone_inference import infer_missing_joints

# ─── 1) Define categories in terms of MediaPipe indices ──────────────
LEFT_ARM = [
    (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_ELBOW.value),
    (PoseLandmark.LEFT_ELBOW.value,    PoseLandmark.LEFT_WRIST.value),
]
RIGHT_ARM = [
    (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_ELBOW.value),
    (PoseLandmark.RIGHT_ELBOW.value,    PoseLandmark.RIGHT_WRIST.value),
]
LEFT_LEG = [
    (PoseLandmark.LEFT_HIP.value,   PoseLandmark.LEFT_KNEE.value),
    (PoseLandmark.LEFT_KNEE.value,  PoseLandmark.LEFT_ANKLE.value),
]
RIGHT_LEG = [
    (PoseLandmark.RIGHT_HIP.value,  PoseLandmark.RIGHT_KNEE.value),
    (PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value),
]
TORSO = [
    (PoseLandmark.LEFT_HIP.value,      PoseLandmark.RIGHT_HIP.value),
    (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value),
    (PoseLandmark.LEFT_HIP.value,      PoseLandmark.LEFT_SHOULDER.value),
    (PoseLandmark.RIGHT_HIP.value,     PoseLandmark.RIGHT_SHOULDER.value),
]
def bidir(lst):
    return set(lst) | {(j,i) for i,j in lst}

CATEGORIES = {
    "Left Arm":  bidir(LEFT_ARM),
    "Right Arm": bidir(RIGHT_ARM),
    "Left Leg":  bidir(LEFT_LEG),
    "Right Leg": bidir(RIGHT_LEG),
    "Torso":     bidir(TORSO),
}

CATEGORY_COLORS = {
    "Left Arm":  (  0,   0, 255),  # Red
    "Right Arm": (255,   0,   0),  # Blue
    "Left Leg":  (  0, 255,   0),  # Green
    "Right Leg": (  0, 255, 255),  # Yellow
    "Torso":     (255,   0, 255),  # Magenta
}

# ─── Drawing params ────────────────────────────────────────────────
JOINT_COLOR    = (0,0,255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 4

def main():
    p = argparse.ArgumentParser(
        description="Grouped overlay + missing‐bone inference"
    )
    p.add_argument("image", help="Input image path")
    p.add_argument("-o","--output", default="overlay_filled.jpg",
                   help="Where to save")
    p.add_argument("--conf", type=float, default=0.1,
                   help="Visibility threshold")
    args = p.parse_args()

    # 1) Load image & MP pose
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read '{args.image}'")
    H, W = frame.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        print("No pose detected."); return

    # 2) Build raw 28×3 array: [x_px, y_px, visibility]
    pose28 = np.zeros((28,3), dtype=np.float32)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None: continue
        lm = res.pose_landmarks.landmark[mp_i]
        pose28[mpi_i] = [lm.x * W, lm.y * H, lm.visibility]


    # 3) Original masks
    jmask = pose28[:,2] > args.conf
    bmask = np.array([jmask[i] and jmask[j] for i,j in MPIINF_EDGES], bool)

    # 4) Fill in missing joints
    pose28_filled = infer_missing_joints(pose28, jmask)

    # 5) New masks on filled pose
    jmask2 = pose28_filled[:,2] > 0
    bmask2 = np.array([jmask2[i] and jmask2[j] for i,j in MPIINF_EDGES], bool)

    # 6) Draw grouped overlay using filled coords
    out = frame.copy()
    # For each category, draw only its edges
    for cat, edges in CATEGORIES.items():
        color = CATEGORY_COLORS[cat]
        for (i_mp, j_mp) in edges:
            # map MP idx → MPIINF idx
            i = MEDIAPIPE_TO_MPIINF.get(i_mp)
            j = MEDIAPIPE_TO_MPIINF.get(j_mp)
            if i is None or j is None: 
                continue
            # skip if neither visible
            if not (jmask2[i] or jmask2[j]):
                continue
            # draw bone
            if jmask2[i] and jmask2[j]:
                thickness = BONE_THICKNESS
            else:
                thickness = BONE_THICKNESS // 2
            p1 = tuple(pose28_filled[i,:2].astype(int))
            p2 = tuple(pose28_filled[j,:2].astype(int))
            cv2.line(out, p1, p2, color, thickness)

    # 7) Draw filled joints
    for k, (x,y,conf) in enumerate(pose28_filled):
        if jmask2[k]:
            cv2.circle(out, (int(x),int(y)), JOINT_RADIUS, JOINT_COLOR, -1)

    # 8) Legend (same as before)
    margin, entry_h = 10, 25
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    max_w = max(cv2.getTextSize(cat, font, fs, th)[0][0] for cat in CATEGORIES)
    box_w = margin*2 + 20 + 5 + max_w
    box_h = margin*2 + entry_h * len(CATEGORIES)
    overlay = out.copy()
    cv2.rectangle(overlay, (0,0),(box_w,box_h),(0,0,0),-1)
    out = cv2.addWeighted(overlay,0.6,out,0.4,0)
    y = margin + entry_h - 5
    for cat, color in CATEGORY_COLORS.items():
        cv2.rectangle(out, (margin, y-entry_h+5),
                      (margin+20, y+5), color, -1)
        cv2.putText(out, cat, (margin+25, y),
                    font, fs, (255,255,255), th, cv2.LINE_AA)
        y += entry_h

    # 9) Save
    cv2.imwrite(args.output, out)
    print("✅ Saved filled grouped overlay to", args.output)

if __name__=="__main__":
    main()
