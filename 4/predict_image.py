# predict_image.py
import cv2
import mediapipe as mp

from visualize import draw_2d_pose

# === 1) Initialize MediaPipe Pose for 2D landmarks ===
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# === 2) Read & preprocess the input image ===
image_path = "input.jpeg"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"❌ Could not find image at: {image_path}")

# Resize to 480×480 so the 2D landmarks map correctly
frame      = cv2.resize(frame, (480, 480))
frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === 3) Run MediaPipe to detect 2D pose ===
results = pose.process(frame_rgb)

# If no landmarks are found, print a message and exit
if not results.pose_landmarks:
    print("❌ No pose detected. Try a different image or higher resolution.")
    exit()

# === 4) Draw the 2D skeleton on top of the frame ===
# 'results.pose_landmarks.landmark' is a list of 33 normalized (x,y,z,visibility).
# We only need x,y for 2D drawing.
output_img = draw_2d_pose(frame.copy(), results.pose_landmarks.landmark)

# === 5) Show & save the final image ===
cv2.imshow("MediaPipe 2D Pose Overlay", output_img)
cv2.imwrite("output.jpg", output_img)
print("✅ Saved output to output.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()
