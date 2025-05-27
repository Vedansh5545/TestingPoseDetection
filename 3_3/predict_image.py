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
image_path = "input.jpeg"  # change if needed
frame = cv2.imread(image_path)
h, w = frame.shape[:2]
size = min(h, w)
centered_frame = frame[(h - size) // 2:(h + size) // 2, (w - size) // 2:(w + size) // 2]
frame_rgb = cv2.cvtColor(centered_frame, cv2.COLOR_BGR2RGB)

# === Detect 2D keypoints
results = pose.process(frame_rgb)

# === Create edge index
def create_edge_index(num_joints=28):
    edges = [[i, i+1] for i in range(num_joints - 1)]
    edges += [list(reversed(e)) for e in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

edge_index = create_edge_index().to(device)

# === If keypoints detected
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

    # === Prepare input
    input_tensor = torch.tensor(np.concatenate([normed_kps, np.zeros((28,1))], axis=1), dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out_3d = model(input_tensor, edge_index.unsqueeze(0)).squeeze(0).cpu().numpy()

    # === Draw and show/save output
    draw_3d_pose(centered_frame, out_3d, image_dims=(centered_frame.shape[1], centered_frame.shape[0]), draw_head=True)

    cv2.imshow("3D Pose Output", centered_frame)
    cv2.imwrite("output.jpg", centered_frame)
    print("✅ Saved output to output.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No person detected in the image.")
