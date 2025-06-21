# batch_detect_and_lift.py

import os
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from visualize import MPIINF_EDGES
from detect_and_lift import detect_and_lift_frame

def save_3d_snapshot(pose3d, out_path, elev=20, azim=-70):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, j in MPIINF_EDGES:
        xs = [pose3d[i, 0], pose3d[j, 0]]
        ys = [pose3d[i, 1], pose3d[j, 1]]
        zs = [pose3d[i, 2], pose3d[j, 2]]
        ax.plot(xs, ys, zs, 'bo-', linewidth=2)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(
        description="Batch-run detect_and_lift over a folder"
    )
    p.add_argument("input_dir", help="Folder with .jpg/.png images")
    p.add_argument("output_dir", help="Folder to save results")
    p.add_argument("-w","--weights", required=True, help="best_model_weights.pth")
    p.add_argument("-m","--meanstd", default="pose2d_mean_std.npy", help="mean/std .npy")
    p.add_argument("--fx", type=float, required=True, help="focal length x")
    p.add_argument("--fy", type=float, required=True, help="focal length y")
    p.add_argument("--cx", type=float, required=True, help="principal point x")
    p.add_argument("--cy", type=float, required=True, help="principal point y")
    p.add_argument("--elev", type=float, default=20, help="3D elev angle")
    p.add_argument("--azim", type=float, default=-70, help="3D azim angle")
    args = p.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(inp.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        overlay2d, pose3d = detect_and_lift_frame(
            str(img_path),
            args.weights,
            args.meanstd,
            args.fx, args.fy, args.cx, args.cy
        )

        # save 2D overlay
        out2d = out / f"{img_path.stem}_2d.png"
        cv2.imwrite(str(out2d), overlay2d)

        # save 3D snapshot
        out3d = out / f"{img_path.stem}_3d.png"
        save_3d_snapshot(pose3d, str(out3d),
                         elev=args.elev, azim=args.azim)

        print(f"Processed {img_path.name}")

if __name__ == "__main__":
    main()
