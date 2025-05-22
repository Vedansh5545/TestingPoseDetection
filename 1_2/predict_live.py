import cv2
import torch
import mediapipe as mp
import numpy as np
from model import PoseEstimator
from visualize import draw_3d_pose

model = PoseEstimator()
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval().cuda()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

def create_edge_index(num_joints=28):
    edges = []
    for i in range(num_joints - 1):
        edges.append([i, i+1])
    return torch.tensor(edges + [list(reversed(e)) for e in edges], dtype=torch.long).t().contiguous()

edge_index = create_edge_index().cuda()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    size = min(h, w)
    frame = cv2.resize(frame, (size, size))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints_2d = np.array([[lm.x, lm.y] for lm in landmarks])[:28]

        keypoints_2d = (keypoints_2d - keypoints_2d.mean()) / (keypoints_2d.std() + 1e-6)
        input_tensor = torch.tensor(keypoints_2d, dtype=torch.float32).unsqueeze(0).cuda()

        if not torch.isnan(input_tensor).any():
            out_3d = model(input_tensor, edge_index.unsqueeze(0))
            out_3d = out_3d.squeeze(0).detach().cpu().numpy()
            draw_3d_pose(frame, out_3d)

    cv2.imshow('3D Pose Estimation Demo', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
