# overlay_2d_labeled_ml.py

import argparse
import cv2
import numpy as np
import torch
import mediapipe as mp

from skeleton_utils import MPIINF_EDGES, MEDIAPIPE_TO_MPIINF
from bone_label_classifier import BoneClassifier, bone_names

# drawing params
BONE_COLOR     = (0,255,0)
JOINT_COLOR    = (0,0,255)
TEXT_COLOR     = (255,255,255)
JOINT_RADIUS   = 4
BONE_THICKNESS = 2
FONT_SCALE     = 0.4
FONT_THICKNESS = 1
CONF_THRESH    = 0.1

def build_features(pose28):
    """
    Given pose28: (28,3) array [x,y,conf],
    returns an (E,7) array of features for each bone:
      [xi, yi, xj, yj, dx, dy, dist]
    """
    feats = []
    for i, j in MPIINF_EDGES:
        xi, yi, _ = pose28[i]
        xj, yj, _ = pose28[j]
        dx, dy = xj - xi, yj - yi
        dist = np.hypot(dx, dy)
        feats.append([xi, yi, xj, yj, dx, dy, dist])
    return np.array(feats, dtype=np.float32)

def draw_ml_labels(frame, pose28, preds):
    """
    Draws green bones, red joints,
    and the ML-predicted bone names.
    preds: length-E array of class indices into bone_names.
    """
    pts = pose28[:,:2].astype(int)
    # bones + labels
    for k, (i, j) in enumerate(MPIINF_EDGES):
        if pose28[i,2] > CONF_THRESH and pose28[j,2] > CONF_THRESH:
            p1 = tuple(pts[i]); p2 = tuple(pts[j])
            cv2.line(frame, p1, p2, BONE_COLOR, BONE_THICKNESS)
            name = bone_names[preds[k]]
            mx, my = (p1[0]+p2[0])//2 + 5, (p1[1]+p2[1])//2 - 5
            cv2.putText(frame, name.replace("_"," "), (mx,my),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    # joints
    for x,y,conf in pose28:
        if conf > CONF_THRESH:
            cv2.circle(frame, (int(x),int(y)), JOINT_RADIUS, JOINT_COLOR, -1)
    return frame

def main():
    parser = argparse.ArgumentParser(
        description="Overlay 2D pose with ML‐predicted bone labels"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-m","--model",
                        default="bone_classifier.pth",
                        help="Path to bone_classifier.pth")
    parser.add_argument("-o","--output",
                        default="overlay_ml.jpg",
                        help="Where to save the output image")
    parser.add_argument("--conf", type=float, default=CONF_THRESH,
                        help="Visibility threshold")
    args = parser.parse_args()

    # 1) Read & detect
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read '{args.image}'")
    H, W = frame.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        print("No pose detected."); return

    # 2) Build 28×3 pose array
    pose28 = np.zeros((28,3), dtype=np.float32)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None: continue
        lm = res.pose_landmarks.landmark[mp_i]
        pose28[mpi_i] = [lm.x*W, (1-lm.y)*H, lm.visibility]

    # 3) Build features + load classifier
    feats = build_features(pose28)
    feats_t = torch.from_numpy(feats)
    clf = BoneClassifier(in_dim=7,
                         hidden_dim=64,
                         num_classes=len(bone_names))
    clf.load_state_dict(torch.load(args.model, map_location="cpu"))
    clf.eval()

    # 4) Predict and draw
    with torch.no_grad():
        logits = clf(feats_t)
        preds = logits.argmax(dim=1).numpy()
    vis = draw_ml_labels(frame.copy(), pose28, preds)

    # 5) Save
    cv2.imwrite(args.output, vis)
    print("✅ Saved ML‐labeled overlay to", args.output)

if __name__=="__main__":
    main()
