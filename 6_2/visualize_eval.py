# visualize_eval.py

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from data_loader import MPIINF3DHPDataset
from model import PoseEstimator, create_edge_index, device
from visualize import draw_2d_pose_28  # use visualize.py


def main():
    # Load dataset (with normalization)
    ds = MPIINF3DHPDataset("mpi_inf_combined.npz", normalize=True)
    n_val = int(0.1 * len(ds))
    _, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    # Load trained model
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
    model.eval()
    edge_index = create_edge_index().to(device)

    # Visualize first 5 validation samples
    for idx in range(min(5, len(val_ds))):
        sample = val_ds[idx]                 # sample is a dict
        x2d = sample['pose2d']              # Tensor (28,3)
        y3d_gt = sample['pose3d']           # Tensor (28,3)
        x2d_np = x2d.cpu().numpy()

        # 2D overlay
        img = np.zeros((512,512,3), np.uint8)
        vis2d = draw_2d_pose_28(img.copy(), x2d_np)
        plt.figure(figsize=(4,4))
        plt.title(f"Sample {idx} — 2D Keypoints")
        plt.axis('off')
        plt.imshow(cv2.cvtColor(vis2d, cv2.COLOR_BGR2RGB))

        # 3D prediction
        with torch.no_grad():
            inp = x2d.unsqueeze(0).to(device)                # [1,28,3]
            # replicate edges for batch dimension
            edge_b = edge_index.repeat(1, inp.size(0)).view(2, -1)
            pred3d = model(inp, edge_b).squeeze(0).cpu().numpy()  # [28,3]

        # 3D plot
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Sample {idx} — Predicted 3D")
        edges = create_edge_index().t().cpu().numpy().reshape(-1,2)
        for i, j in edges:
            if i < j:
                ax.plot(
                    [pred3d[i,0], pred3d[j,0]],
                    [pred3d[i,1], pred3d[j,1]],
                    [pred3d[i,2], pred3d[j,2]], 'bo-'
                )
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-70)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
