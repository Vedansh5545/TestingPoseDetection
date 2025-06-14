# predict_image.py

import argparse
import cv2
import numpy as np
import mediapipe as mp

from skeleton_utils import MEDIAPIPE_TO_MPIINF
from visualize_28 import draw_2d_pose_28

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_image", help="Path to input image file")
    p.add_argument("--output_img", default="output_28.jpg", help="Path to save visualized 2D pose")
    p.add_argument("--output_npy", default="pose2d_28.npy", help="Path to save (28,3) numpy array")
    args = p.parse_args()

    # read
    frame = cv2.imread(args.input_image)
    if frame is None:
        raise FileNotFoundError(f"Could not read {args.input_image}")
    H, W = frame.shape[:2]

    # init MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    # process
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        raise RuntimeError("No pose detected")

    # build 28×3 array
    pose2d_28 = np.zeros((28, 3), dtype=np.float32)
    for mp_idx, mpi_idx in MEDIAPIPE_TO_MPIINF.items():
        if mpi_idx is None:
            continue
        lm = results.pose_landmarks.landmark[mp_idx]
        pose2d_28[mpi_idx] = [lm.x * W, lm.y * H, lm.visibility]

    # save
    np.save(args.output_npy, pose2d_28)
    print(f"Saved 2D pose to {args.output_npy}")

    # visualize & save
    vis = draw_2d_pose_28(frame.copy(), pose2d_28)
    cv2.imwrite(args.output_img, vis)
    print(f"Saved visualization to {args.output_img}")

if __name__ == "__main__":
    main()
