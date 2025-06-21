# pipeline.py

import argparse
import cv2
import numpy as np
import torch
import mediapipe as mp

from skeleton_utils import MEDIAPIPE_TO_MPIINF, MPIINF_EDGES
from missing_bone_inference import infer_missing_joints
from model import PoseEstimator, create_edge_index, device
from overlay_2d_grouped_filled import CATEGORIES, CATEGORY_COLORS

# Load normalization statistics
mean2d, std2d = np.load("pose2d_mean_std.npy")  # shape (3,)

# Load and prepare model
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
model.eval()
edge_index = create_edge_index().to(device)

# Simple camera intrinsics for SolvePnP (adjust as needed)
K = None  # If you have intrinsics, set K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])


def run_pipeline(image_path: str, output_image: str = "overlay_final.jpg"):
    # 1) Read image & detect 2D
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read '{image_path}'")
    H, W = img.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        raise RuntimeError("No pose detected.")

    # 2) Build & fill pose28
    pose28 = np.zeros((28,3), dtype=float)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None:
            continue
        lm = res.pose_landmarks.landmark[mp_i]
        pose28[mpi_i] = [lm.x * W, lm.y * H, lm.visibility]
    jmask = pose28[:,2] > 0.1
    pose28_filled = infer_missing_joints(pose28, jmask)

    # 3) Normalize & prepare model input
    xy = pose28_filled[:,:2]
    xy_norm = (xy - mean2d[:2]) / std2d[:2]
    zeros = np.zeros((28,1), dtype=float)
    inp = np.concatenate([xy_norm, zeros], axis=1)
    x = torch.from_numpy(inp).float().unsqueeze(0).to(device)

    # 4) Predict 3D
    with torch.no_grad():
        pred3d = model(x, edge_index).cpu().numpy()[0]

    # 5) Optionally solvePnP if K provided
    overlay = img.copy()
    if K is not None:
        # use full perspective if intrinsics known
        _, rvec, tvec = cv2.solvePnP(
            pred3d.astype(np.float32),
            pose28_filled[:,:2].astype(np.float32),
            K, None
        )
        proj, _ = cv2.projectPoints(pred3d.astype(np.float32), rvec, tvec, K, None)
        proj2d = proj.reshape(-1,2)
    else:
        # simple orthographic: drop Z
        proj2d = pose28_filled[:,:2]

    # 6) Draw grouped overlay using proj2d
    for cat, edges in CATEGORIES.items():
        color = CATEGORY_COLORS[cat]
        for (i_mp, j_mp) in edges:
            i = MEDIAPIPE_TO_MPIINF.get(i_mp)
            j = MEDIAPIPE_TO_MPIINF.get(j_mp)
            if i is None or j is None:
                continue
            p1 = tuple(proj2d[i].astype(int))
            p2 = tuple(proj2d[j].astype(int))
            cv2.line(overlay, p1, p2, color, 3)
    for x,y,_ in pose28_filled:
        cv2.circle(overlay, (int(x),int(y)), 4, (0,0,255), -1)

    # 7) Save
    cv2.imwrite(output_image, overlay)
    print(f"✅ Saved pipeline overlay to '{output_image}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 2D→3D pipeline with overlay")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o","--output", default="overlay_final.jpg",
                        help="Output overlay image")
    args = parser.parse_args()
    run_pipeline(args.image, args.output)
