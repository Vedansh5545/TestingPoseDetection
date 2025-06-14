# infer.py
import argparse
import numpy as np
import torch
import cv2
import mediapipe as mp
from model import PoseEstimator, create_edge_index
from skeleton_utils import MEDIAPIPE_TO_COCO17
from visualize import draw_coco17, plot_3d

def extract_2d(image_path):
    frame = cv2.imread(image_path)
    H,W = frame.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError("No person")
    arr = np.zeros((17,3),dtype=np.float32)
    for mp_i, coco_i in MEDIAPIPE_TO_COCO17.items():
        if coco_i is None: continue
        lm = res.pose_landmarks.landmark[mp_i]
        x,y,c = lm.x*W, (1-lm.y)*H, lm.visibility
        arr[coco_i] = [x,y,c]
    return frame, arr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--weights","-w",default="best_model_weights.pth")
    args = p.parse_args()

    frame, p2d = extract_2d(args.image)
    np.save("pose2d_coco17.npy", p2d)
    vis2d = draw_coco17(frame.copy(), p2d)
    cv2.imwrite("overlay_coco17.jpg", vis2d)

    # 3D
    x = torch.tensor(p2d, dtype=torch.float32).unsqueeze(0)
    edge = create_edge_index().to(x.device)
    model = PoseEstimator().to(x.device)
    model.load_state_dict(torch.load(args.weights,map_location=x.device))
    model.eval()
    with torch.no_grad():
        pred3d = model(x, edge).squeeze(0).cpu().numpy()

    plot_3d(pred3d)

if __name__=="__main__":
    main()
