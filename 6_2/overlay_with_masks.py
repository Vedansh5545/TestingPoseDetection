# overlay_with_masks.py

import argparse
import cv2
import numpy as np
import mediapipe as mp

from skeleton_utils import MPIINF_EDGES, MEDIAPIPE_TO_MPIINF

# ─── Drawing parameters ────────────────────────────────────────
VIS_BONE_COLOR = (0, 255, 0)     # both endpoints visible
HID_BONE_COLOR = (128, 128, 128) # one endpoint visible
JOINT_COLOR    = (0, 0, 255)     # visible joints
JOINT_RADIUS   = 4
BONE_THICKNESS = 2
CONF_THRESH    = 0.1             # confidence threshold

def draw_with_masks(frame: np.ndarray,
                    pose28: np.ndarray,
                    jmask: np.ndarray,
                    bmask: np.ndarray) -> np.ndarray:
    """
    Draw bones and joints with visibility masks:
      - green if bone fully visible,
      - gray if bone partially visible,
      - skip if bone completely invisible,
      - red dots for visible joints.
    """
    pts = pose28[:, :2].astype(int)

    # Draw bones
    for idx, (i, j) in enumerate(MPIINF_EDGES):
        # skip if neither endpoint is visible
        if not (jmask[i] or jmask[j]):
            continue

        # choose color: green if both visible else gray
        color = VIS_BONE_COLOR if bmask[idx] else HID_BONE_COLOR
        p1 = tuple(pts[i])
        p2 = tuple(pts[j])
        cv2.line(frame, p1, p2, color, BONE_THICKNESS)

    # Draw joints
    for k, (x, y, conf) in enumerate(pose28):
        if jmask[k]:
            cv2.circle(frame, (int(x), int(y)), JOINT_RADIUS, JOINT_COLOR, -1)

    return frame

def main():
    parser = argparse.ArgumentParser(
        description="2D overlay with visibility masks"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output", default="overlay_masks.jpg",
                        help="Where to save the resulting image")
    parser.add_argument("--conf", type=float, default=CONF_THRESH,
                        help="Visibility threshold")
    args = parser.parse_args()

    # 1) Load image
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read '{args.image}'")
    H, W = frame.shape[:2]

    # 2) Run MediaPipe
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        print("No person detected.")
        return

    # 3) Build 28×3 pose array [x_px, y_px, visibility]
    pose28 = np.zeros((28, 3), dtype=np.float32)
    for mp_i, mpi_i in MEDIAPIPE_TO_MPIINF.items():
        if mpi_i is None:
            continue
        lm = res.pose_landmarks.landmark[mp_i]
        x_px = lm.x * W
        y_px = (1 - lm.y) * H  # flip so image-up → plot-up
        pose28[mpi_i] = [x_px, y_px, lm.visibility]

    # 4) Compute joint & bone masks
    jmask = pose28[:, 2] > args.conf
    # bone mask: True if both joints visible
    bmask = np.array([jmask[i] and jmask[j] for i, j in MPIINF_EDGES], dtype=bool)

    # 5) Draw & save
    vis = draw_with_masks(frame.copy(), pose28, jmask, bmask)
    cv2.imwrite(args.output, vis)
    print("✅ Saved overlay with masks to", args.output)

if __name__ == "__main__":
    main()
