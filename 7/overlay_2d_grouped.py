# overlay_2d_grouped.py

import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark

# ─── 1) Define categories and their MediaPipe connections ─────────────
LEFT_ARM = [
    (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_ELBOW.value),
    (PoseLandmark.LEFT_ELBOW.value,   PoseLandmark.LEFT_WRIST.value),
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

# Make each category bidirectional
def bidir(lst):
    return set(lst) | set((j,i) for i,j in lst)

CATEGORIES = {
    "Left Arm":    bidir(LEFT_ARM),
    "Right Arm":   bidir(RIGHT_ARM),
    "Left Leg":    bidir(LEFT_LEG),
    "Right Leg":   bidir(RIGHT_LEG),
    "Torso":       bidir(TORSO),
}

# Assign a distinct BGR color to each category
CATEGORY_COLORS = {
    "Left Arm":  (  0,   0, 255),  # Red
    "Right Arm": (255,   0,   0),  # Blue
    "Left Leg":  (  0, 255,   0),  # Green
    "Right Leg": (  0, 255, 255),  # Yellow
    "Torso":     (255,   0, 255),  # Magenta
}

# Drawing params
JOINT_COLOR    = (0,0,255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 4
CONF_THRESH    = 0.1

def draw_grouped_overlay(image_path, output_path, conf_thresh):
    # 1) Load & detect
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read '{image_path}'")
    H, W = frame.shape[:2]

    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        print("No pose detected."); return

    lms = res.pose_landmarks.landmark
    # 2) Draw bones by category
    for (i,j) in mp.solutions.pose.POSE_CONNECTIONS:
        lm1, lm2 = lms[i], lms[j]
        if lm1.visibility < conf_thresh or lm2.visibility < conf_thresh:
            continue
        p1 = (int(lm1.x*W), int(lm1.y*H))
        p2 = (int(lm2.x*W), int(lm2.y*H))
        # find category
        for cat, edges in CATEGORIES.items():
            if (i,j) in edges:
                color = CATEGORY_COLORS[cat]
                cv2.line(frame, p1, p2, color, BONE_THICKNESS)
                break
        else:
            # skip unlabeled bones
            continue

    # 3) Draw joints
    for lm in lms:
        if lm.visibility < conf_thresh:
            continue
        x,y = int(lm.x*W), int(lm.y*H)
        cv2.circle(frame, (x,y), JOINT_RADIUS, JOINT_COLOR, -1)

    # 4) Draw legend
    margin = 10
    entry_h = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 2
    # calculate box size
    max_w = 0
    for cat in CATEGORIES:
        w,_ = cv2.getTextSize(cat, font, fs, th)[0]
        max_w = max(max_w, w)
    box_w = margin*2 + 20 + 5 + max_w
    box_h = margin*2 + entry_h * len(CATEGORIES)

    # semi-transparent bg
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (box_w, box_h), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # entries
    y = margin + entry_h - 5
    for cat, color in CATEGORY_COLORS.items():
        # color square
        cv2.rectangle(frame, (margin, y-entry_h+5),
                      (margin+20, y+5), color, -1)
        # text
        cv2.putText(frame, cat,
                    (margin+25, y),
                    font, fs, (255,255,255), th, cv2.LINE_AA)
        y += entry_h

    # 5) Save
    cv2.imwrite(output_path, frame)
    print(f"✅ Saved grouped overlay to '{output_path}'")

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="2D pose overlay colored by body part with legend"
    )
    p.add_argument("image", help="Input image")
    p.add_argument("-o","--output", default="overlay_grouped.jpg",
                   help="Where to save")
    p.add_argument("--conf", type=float, default=CONF_THRESH,
                   help="Visibility threshold")
    args = p.parse_args()
    draw_grouped_overlay(args.image, args.output, args.conf)
