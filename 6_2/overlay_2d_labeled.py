# overlay_2d_labeled.py

import argparse
import cv2
import numpy as np
import mediapipe as mp

from skeleton_utils import MEDIAPIPE_TO_MPIINF
from visualize_labeled import draw_2d_pose_with_labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Input image path")
    p.add_argument("-o","--output", default="overlay_2d_labeled.jpg",
                   help="Where to save the output")
    p.add_argument("--conf", type=float, default=0.1,
                   help="Visibility threshold")
    args = p.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read {args.image}")
    H, W = frame.shape[:2]

    # MediaPipe 2D detection
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            print("No pose detected.")
            return

    # Build 28×3 array
    pose2d = np.zeros((28,3), dtype=np.float32)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None: continue
        lm = res.pose_landmarks.landmark[mp_i]
        pose2d[mpi_i] = [lm.x*W, (1-lm.y)*H, lm.visibility]

    # Draw & save
    vis = draw_2d_pose_with_labels(frame.copy(), pose2d)
    cv2.imwrite(args.output, vis)
    print("Saved labeled overlay to", args.output)

if __name__=="__main__":
    main()
