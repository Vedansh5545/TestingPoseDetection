# detect_and_lift.py

import cv2
import numpy as np
import torch
import mediapipe as mp

from skeleton_utils import MEDIAPIPE_TO_MPIINF
from model import PoseEstimator, create_edge_index, device
from visualize import draw_2d_pose_28

def detect_and_lift_frame(
    image_path: str,
    weights_path: str,
    meanstd_path: str
):
    """
    1) Detect 2D with MediaPipe
    2) Build 28×3 [x_px, y_px (flipped), conf]
    3) Standardize exactly as in training
    4) Predict 3D with your trained model
    Returns:
      - overlay2d: BGR image with 2D skeleton
      - pose3d: (28,3) numpy array in mm
    """
    # 1) Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = frame.shape[:2]

    # 2) MediaPipe 2D detector
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError("No pose detected")

    # 3) Remap to 28 joints, with y flipped so “up” is positive
    pose2d = np.zeros((28,3), dtype=np.float32)
    for mp_idx, mpi_idx in MEDIAPIPE_TO_MPIINF.items():
        if mpi_idx is None:
            continue
        lm = res.pose_landmarks.landmark[mp_idx]
        x_px = lm.x * W
        y_px = (1 - lm.y) * H      # **flip** so image-up maps to +Y
        pose2d[mpi_idx] = [x_px, y_px, lm.visibility]

    # 4) Draw & return the 2D overlay
    overlay2d = draw_2d_pose_28(frame.copy(), pose2d)

    # 5) Standardize exactly as in training (raw pixels → mean/std)
    mean, std = np.load(meanstd_path)  # from your data_loader np.save :contentReference[oaicite:0]{index=0}
    inp = (pose2d - mean) / std
    x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)  # (1,28,3)

    # 6) Load model & predict 3D
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    edge_index = create_edge_index().to(device)
    with torch.no_grad():
        pred3d = model(x, edge_index).squeeze(0).cpu().numpy()  # (28,3) in mm

    return overlay2d, pred3d


if __name__ == "__main__":
    import argparse
    from visualize import plot_3d

    p = argparse.ArgumentParser(
        description="Detect 2D → standardize → lift to 3D"
    )
    p.add_argument("image", help="Path to input image")
    p.add_argument("-w","--weights", default="best_model_weights.pth",
                   help="Your trained PoseEstimator .pth")
    p.add_argument("-m","--meanstd", default="pose2d_mean_std.npy",
                   help="Mean/std file from data prep")
    args = p.parse_args()

    overlay2d, pose3d = detect_and_lift_frame(
        args.image, args.weights, args.meanstd
    )

    # Save the 2D overlay
    out2d = "overlay2d_corrected.png"
    cv2.imwrite(out2d, overlay2d)
    print(f"Saved corrected 2D overlay → {out2d}")

    # Plot the 3D skeleton
    plot_3d(pose3d, elev=20, azim=-70)
