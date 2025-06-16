# overlay_2d_colored_legend.py

import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark

def generate_distinct_bgr_colors(n):
    """
    Generate n visually distinct colors in BGR by sampling HSV hues.
    """
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # Hue angle in [0,179]
        # full saturation and value
        hsv_pixel = np.uint8([[[hue, 255, 255]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0,0]
        colors.append((int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])))
    return colors

def draw_colored_overlay_with_legend(image_path: str,
                                     output_path: str,
                                     conf_thresh: float = 0.1):
    # --- 1) Read image and run MediaPipe Pose ---
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read '{image_path}'")
    H, W = frame.shape[:2]

    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        print("No pose detected.")
        return

    landmarks = results.pose_landmarks.landmark

    # --- 2) Prepare connections and colors ---
    connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    connections.sort()
    N = len(connections)
    colors = generate_distinct_bgr_colors(N)

    # --- 3) Draw bones in unique colors ---
    for idx, (start_idx, end_idx) in enumerate(connections):
        lm1 = landmarks[start_idx]
        lm2 = landmarks[end_idx]
        if lm1.visibility < conf_thresh or lm2.visibility < conf_thresh:
            continue
        x1, y1 = int(lm1.x * W), int(lm1.y * H)
        x2, y2 = int(lm2.x * W), int(lm2.y * H)
        cv2.line(frame, (x1, y1), (x2, y2), colors[idx], 2)

    # --- 4) Draw joints as red dots ---
    for lm in landmarks:
        if lm.visibility < conf_thresh:
            continue
        x, y = int(lm.x * W), int(lm.y * H)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # --- 5) Build legend entries ---
    legend_entries = []
    for idx, (start_idx, end_idx) in enumerate(connections):
        name1 = PoseLandmark(start_idx).name.lower()
        name2 = PoseLandmark(end_idx).name.lower()
        bone_name = f"{name1}→{name2}"
        legend_entries.append((colors[idx], bone_name))

    # --- 6) Overlay legend in top-left corner ---
    # parameters
    entry_height = 20
    margin = 10
    text_offset = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    # compute legend box size
    max_text_width = 0
    for _, name in legend_entries:
        (w, _), _ = cv2.getTextSize(name, font, font_scale, thickness)
        max_text_width = max(max_text_width, w)
    box_width = margin + 16 + text_offset + max_text_width + margin
    box_height = margin + entry_height * len(legend_entries) + margin

    # draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0,0,0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # draw each legend entry
    y = margin + entry_height - 6
    for color, name in legend_entries:
        # colored square
        cv2.rectangle(frame,
                      (margin, y - entry_height + 6),
                      (margin + 16, y - entry_height + 6 + 12),
                      color, -1)
        # text
        cv2.putText(frame, name,
                    (margin + 16 + text_offset, y),
                    font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        y += entry_height

    # --- 7) Save output ---
    cv2.imwrite(output_path, frame)
    print(f"✅ Saved colored overlay with legend to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D pose overlay with each bone in a distinct color and a legend"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o","--output", default="overlay_colored_legend.jpg",
                        help="Output path for the overlay")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Visibility threshold for joints")
    args = parser.parse_args()

    draw_colored_overlay_with_legend(args.image, args.output, args.conf)
