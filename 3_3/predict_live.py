import cv2
import torch
import numpy as np
import mediapipe as mp
from model import PoseEstimator, create_edge_index, device

model = PoseEstimator().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

pose2d_stats = np.load("pose2d_mean_std.npy", allow_pickle=True)
pose2d_mean, pose2d_std = pose2d_stats[0][:2], pose2d_stats[1][:2]
edge_index = create_edge_index().to(device)

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    size = min(h, w)
    crop = frame[(h-size)//2:(h+size)//2, (w-size)//2:(w+size)//2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)

    if results.pose_landmarks and len(results.pose_landmarks.landmark) >= 28:
        keypoints_2d = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark[:28]], dtype=np.float32)
        norm = (keypoints_2d - pose2d_mean) / (pose2d_std + 1e-6)
        keypoints_3d = np.concatenate([norm, np.zeros((28,1), dtype=np.float32)], axis=-1)
        input_tensor = torch.tensor(keypoints_3d, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            model(input_tensor, edge_index.unsqueeze(0))

        for (x, y) in (keypoints_2d * [crop.shape[1], crop.shape[0]]).astype(int):
            cv2.circle(crop, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow("Live Pose Detection", crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
