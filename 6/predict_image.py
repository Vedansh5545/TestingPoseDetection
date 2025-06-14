# predict_image.py

import argparse
import cv2
import numpy as np
import mediapipe as mp
from skeleton_utils import MEDIAPIPE_TO_MPIINF
from visualize import draw_2d_pose_28

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_image", help="Path to input image")
    p.add_argument("--output_img", default="overlay_28.jpg",
                   help="Where to save the 2D overlay")
    p.add_argument("--output_npy", default="pose2d_mpi.npy",
                   help="Where to save the (28,3) numpy array")
    args = p.parse_args()

    frame = cv2.imread(args.input_image)
    if frame is None:
        raise FileNotFoundError(f"Could not read {args.input_image}")
    H, W = frame.shape[:2]

    # MediaPipe single-frame pose
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError("No pose detected")

    # Initialize 28×3 pose array
    pose2d = np.zeros((28,3), dtype=np.float32)

    # Map in all direct landmarks
    for mp_idx, mpi_idx in MEDIAPIPE_TO_MPIINF.items():
        if mpi_idx is None:
            continue
        lm = res.pose_landmarks.landmark[mp_idx]
        x_px = lm.x * W
        y_px = (1 - lm.y) * H   # flip Y so image-up → plot-up
        pose2d[mpi_idx] = [x_px, y_px, lm.visibility]

    # Derive pelvis (idx 0) as midpoint of hips (1 & 5)
    lh = pose2d[5,:2]; rh = pose2d[1,:2]
    vh = (pose2d[5,2] > 0) and (pose2d[1,2] > 0)
    pose2d[0] = [*((lh+rh)/2), vh * 0.5]

    # Derive neck (idx 10) as midpoint of shoulders (12 & 15)
    ls = pose2d[12,:2]; rs = pose2d[15,:2]
    vs = (pose2d[12,2] > 0) and (pose2d[15,2] > 0)
    pose2d[10] = [*((ls+rs)/2), vs * 0.5]

    # Save and visualize
    np.save(args.output_npy, pose2d)
    print("Saved 2D pose to", args.output_npy)

    vis = draw_2d_pose_28(frame.copy(), pose2d)
    cv2.imwrite(args.output_img, vis)
    print("Saved overlay to", args.output_img)

if __name__ == "__main__":
    main()
