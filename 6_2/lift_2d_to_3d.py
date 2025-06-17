# lift_2d_to_3d.py

import argparse
import numpy as np
import torch
import cv2
import mediapipe as mp

from skeleton_utils import MEDIAPIPE_TO_MPIINF
from missing_bone_inference import infer_missing_joints
from model import PoseEstimator, create_edge_index, device

# Load normalization statistics
mean2d, std2d = np.load("pose2d_mean_std.npy")  # shape (3,) for x, y, z

# Initialize model
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()

# Precompute edge index
edge_index = create_edge_index().to(device)


def lift_image(image_path: str, output_npy: str):
    # 1) Read image and detect 2D keypoints
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read '{image_path}'")
    H, W = img.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        raise RuntimeError("No pose detected in the image.")

    # 2) Build raw (28,3) pose28 array
    pose28 = np.zeros((28, 3), dtype=float)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None:
            continue
        lm = results.pose_landmarks.landmark[mp_i]
        pose28[mpi_i] = [lm.x * W, lm.y * H, lm.visibility]

    # 3) Fill missing joints
    jmask = pose28[:, 2] > 0.1
    pose28_filled = infer_missing_joints(pose28, jmask)

    # 4) Normalize XY channels and zero-pad Z
    xy = pose28_filled[:, :2]
    xy_norm = (xy - mean2d[:2]) / std2d[:2]
    zeros = np.zeros((28, 1), dtype=float)
    inp = np.concatenate([xy_norm, zeros], axis=1)  # shape (28,3)

    # 5) Prepare tensor and run model
    x = torch.from_numpy(inp).float().unsqueeze(0).to(device)  # (1,28,3)
    with torch.no_grad():
        pred3d = model(x, edge_index).cpu().numpy()[0]  # (28,3) in mm

    # 6) Save prediction
    np.save(output_npy, pred3d)
    print(f"✅ Saved 3D joints to '{output_npy}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lift 2D pose to 3D and save as .npy")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output", default="pred3d.npy", help="Output .npy file for 3D joints")
    args = parser.parse_args()
    lift_image(args.image, args.output)


