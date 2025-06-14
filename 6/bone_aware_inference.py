# bone_aware_inference.py

import argparse
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from skeleton_utils import MPIINF_EDGES
from model import PoseEstimator, create_edge_index

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_npy", help="The pose2d_mpi.npy file")
    p.add_argument("--weights", default="best_model_weights.pth",
                   help="Trained model weights (.pth)")
    args = p.parse_args()

    # Load 2D pose
    pose2d = np.load(args.input_npy)                # (28,3)
    pose2d_t = torch.tensor(pose2d, dtype=torch.float32).unsqueeze(0)  # (1,28,3)

    # Get edge_index from model.py
    edge_index = create_edge_index().to(pose2d_t.device)  # [2, 2*E]

    # Load your model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Inference
    inp = torch.cat([pose2d_t, torch.ones_like(pose2d_t[...,:1])], dim=-1)  # (1,28,4)
    with torch.no_grad():
        pred3d = model(inp.to(device), edge_index.unsqueeze(0).to(device))\
                      .squeeze(0).cpu().numpy()  # (28,3)

    # Plot 3D skeleton
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    for i,j in MPIINF_EDGES:
        xs = [pred3d[i,0], pred3d[j,0]]
        ys = [pred3d[i,1], pred3d[j,1]]
        zs = [pred3d[i,2], pred3d[j,2]]
        ax.plot(xs, ys, zs, 'bo-', linewidth=2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-70)
    plt.show()

if __name__ == "__main__":
    main()
