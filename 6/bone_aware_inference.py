# bone_aware_inference.py

import argparse
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from model import PoseEstimator    # adjust import to your model’s class
from skeleton_utils import MPIINF_EDGES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_npy", help="Path to pose2d_28.npy")
    p.add_argument("--weights", default="best_model_weights.pth", help="Path to trained .pth")
    args = p.parse_args()

    # load 2D input
    pose2d = np.load(args.input_npy)   # shape (28,3)
    assert pose2d.shape == (28,3)
    pose2d_tensor = torch.tensor(pose2d, dtype=torch.float32).unsqueeze(0)  # (1,28,3)

    # build adjacency
    adj = np.zeros((28,28), dtype=int)
    for i,j in MPIINF_EDGES:
        adj[i,j] = adj[j,i] = 1
    edge_index, _ = dense_to_sparse(torch.tensor(adj))

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # inference
    inp = torch.cat([pose2d_tensor, torch.ones_like(pose2d_tensor[...,:1])], dim=-1)  # (1,28,4)
    with torch.no_grad():
        pred3d = model(inp.to(device), edge_index.to(device)).squeeze(0).cpu().numpy()  # (28,3)

    # plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    for i,j in MPIINF_EDGES:
        x = [pred3d[i,0], pred3d[j,0]]
        y = [pred3d[i,1], pred3d[j,1]]
        z = [pred3d[i,2], pred3d[j,2]]
        ax.plot(x,y,z,'bo-')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-70)
    plt.show()

if __name__ == "__main__":
    main()
