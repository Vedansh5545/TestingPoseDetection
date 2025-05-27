import cv2
import torch
import numpy as np
import mediapipe as mp
from model import PoseEstimator
from visualize import draw_3d_pose  # Assumes this draws skeleton on image

# === Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load trained model
model = PoseEstimator().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# === MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# === Load and center image
image_path = "input.jpeg"  # Update if needed
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"❌ Could not find image at: {image_path}")

h, w = frame.shape[:2]
size = min(h, w)
centered_frame = frame[(h - size) // 2:(h + size) // 2, (w - size) // 2:(w + size) // 2]
frame_rgb = cv2.cvtColor(centered_frame, cv2.COLOR_BGR2RGB)

# === Detect 2D keypoints
results = pose.process(frame_rgb)

# === Define anatomical skeleton
def create_edge_index():
    edges = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 27),
        (0, 15), (0, 16), (15, 17), (16, 18)
    ]
    # Clip to safe indices
    edges = [(i, j) for i, j in edges if i < 28 and j < 28]
    edges += [(j, i) for i, j in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


edge_index = create_edge_index().to(device)

# === Alignment function
def align_pose_to_image(out_3d):
    out_3d[:, 0] *= -1  # Flip X
    theta = np.radians(30)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    out_3d = out_3d @ R_z.T
    if 11 < len(out_3d) and 12 < len(out_3d):
        shoulder_width = np.linalg.norm(out_3d[11] - out_3d[12])
        if shoulder_width > 1e-4:
            scale = 180 / shoulder_width
            out_3d *= scale
    return out_3d

# === Process if person detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    if len(landmarks) < 28:
        print("⚠️ Not enough keypoints detected. Try another image.")
        exit()

    keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks])[:28]

    # Draw 2D keypoints for visual debugging
    for pt in keypoints_2d:
        x = int(pt[0] * size)
        y = int(pt[1] * size)
        cv2.circle(centered_frame, (x, y), 4, (0, 0, 255), -1)

    # === Normalize keypoints
    mean = keypoints_2d.mean(axis=0)
    std = keypoints_2d.std(axis=0)
    normed_kps = (keypoints_2d - mean) / (std + 1e-6)

    # === Prepare 3D input: [x, y, 0.0] for each joint
    input_tensor = torch.tensor(
        np.concatenate([normed_kps, np.zeros((28, 1))], axis=1),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # === Predict 3D pose
    with torch.no_grad():
        out_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

    # === Align to match image pose
    out_3d = align_pose_to_image(out_3d)

    # === Draw and save
    draw_3d_pose(centered_frame, out_3d, image_dims=(centered_frame.shape[1], centered_frame.shape[0]), draw_head=True)
    cv2.imshow("3D Pose Output", centered_frame)
    cv2.imwrite("output.jpg", centered_frame)
    print("✅ Saved output to output.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No person detected in the image.")
