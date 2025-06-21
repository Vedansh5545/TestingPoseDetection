# overlay_2d.py

import argparse
import cv2
import mediapipe as mp

def draw_2d_overlay(image_path: str, output_path: str, conf_thresh: float = 0.1):
    """
    Reads an image, runs MediaPipe pose detection, and draws:
      - Bones as green lines
      - Joints as red dots
    Saves the result to output_path.
    """
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read '{image_path}'")
    H, W = frame.shape[:2]

    # Run MediaPipe Pose
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("No pose detected.")
        return

    # Draw bones (green) for each POSE_CONNECTION if both endpoints are visible
    for (start_idx, end_idx) in mp.solutions.pose.POSE_CONNECTIONS:
        lm_start = results.pose_landmarks.landmark[start_idx]
        lm_end   = results.pose_landmarks.landmark[end_idx]
        if lm_start.visibility < conf_thresh or lm_end.visibility < conf_thresh:
            continue
        x1, y1 = int(lm_start.x * W), int(lm_start.y * H)
        x2, y2 = int(lm_end.x   * W), int(lm_end.y   * H)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw joints (red)
    for lm in results.pose_landmarks.landmark:
        if lm.visibility < conf_thresh:
            continue
        x, y = int(lm.x * W), int(lm.y * H)
        cv2.circle(frame, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

    # Save output
    cv2.imwrite(output_path, frame)
    print(f"Saved 2D overlay to '{output_path}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Draw 2D pose overlay: bones=green, joints=red."
    )
    p.add_argument("image",    help="Path to input image")
    p.add_argument("-o","--output", default="overlay_2d.jpg",
                   help="Path to save the overlaid image")
    p.add_argument("--conf", type=float, default=0.1,
                   help="Visibility threshold (0–1) for drawing")
    args = p.parse_args()

    draw_2d_overlay(args.image, args.output, args.conf)
