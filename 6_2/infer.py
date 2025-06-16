# infer.py

import argparse
import cv2
import numpy as np
import torch
import mediapipe as mp

from skeleton_utils import MEDIAPIPE_TO_MPIINF
from model import PoseEstimator, create_edge_index, device
from visualize import draw_2d_pose_28, plot_3d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to input image")
    p.add_argument("--weights", "-w", default="best_model_weights.pth",
                   help="Trained model weights")
    p.add_argument("--output_img", "-o", default="overlay_28.jpg",
                   help="Where to save the 2D overlay")
    p.add_argument("--norm_stats", default="pose2d_mean_std.npy",
                   help="(mean,std) for normalizing 2D input")
    args = p.parse_args()

    # --- 1) Read image & detect 2D keypoints ---
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read {args.image}")
    H, W = frame.shape[:2]

    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError("No pose detected")

    # --- 2) Build raw 28×3 array (pixel coords + visibility) ---
    pose2d = np.zeros((28,3), dtype=np.float32)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None:
            continue
        lm = res.pose_landmarks.landmark[mp_i]
        x_px = lm.x * W
        y_px = (1 - lm.y) * H      # flip so image-up → plot-up
        pose2d[mpi_i] = [x_px, y_px, lm.visibility]

    # --- 3) Draw & save 2D overlay ---
    vis2d = draw_2d_pose_28(frame.copy(), pose2d)
    cv2.imwrite(args.output_img, vis2d)
    print("Saved 2D overlay to", args.output_img)

    # --- 4) Normalize exactly as during training ---
    mean, std = np.load(args.norm_stats)  # shape (2,3): [mean, std]
    pose2d_norm = (pose2d - mean) / std

    # --- 5) Prepare tensors & model ---
    x = torch.tensor(pose2d_norm, dtype=torch.float32).unsqueeze(0).to(device)  # [1,28,3]
    edge_index = create_edge_index().to(device)                                # [2,E]
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # --- 6) Predict 3D ---
    with torch.no_grad():
        pred3d = model(x, edge_index).squeeze(0).cpu().numpy()  # (28,3) in mm

    # --- 7) Recenter at pelvis (joint 0) & convert to meters ---
    pelvis = pred3d[0].copy()
    pred3d -= pelvis
    pred3d /= 1000.0  # mm → m

    # --- 8) Plot the human-scaled 3D skeleton ---
    plot_3d(pred3d, elev=20, azim=-70)

if __name__ == "__main__":
    main()
