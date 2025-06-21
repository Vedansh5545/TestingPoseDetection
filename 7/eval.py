# eval.py

import argparse
import torch
from torch.utils.data import DataLoader
from data_loader import MPIINF3DHPDataset
from model import PoseEstimator, create_edge_index, device

def mpjpe(pred, gt):
    """Mean per‐joint position error."""
    return torch.mean(torch.norm(pred - gt, dim=-1))

def main():
    p = argparse.ArgumentParser(
        description="Evaluate a trained PoseEstimator on MPI-INF-3DHP"
    )
    p.add_argument(
        "--data", "-d",
        default="mpi_inf_combined.npz",
        help="Path to the .npz containing 'pose2d' and 'pose3d'"
    )
    p.add_argument(
        "--weights", "-w",
        required=True,
        help="Path to your trained model .pth file"
    )
    p.add_argument(
        "--bs", "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    args = p.parse_args()

    # Load dataset (normalized in __init__ of your Dataset)
    dataset = MPIINF3DHPDataset(args.data, normalize=True)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)

    # Load model
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Prepare graph connectivity
    edge_index = create_edge_index().to(device)

    # Run evaluation
    total_error = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x2d = batch["pose2d"].to(device)  # (B,28,3)
            y3d = batch["pose3d"].to(device)  # (B,28,3)
            pred3d = model(x2d, edge_index)   # (B,28,3)

            batch_error = mpjpe(pred3d, y3d).item() * x2d.size(0)
            total_error += batch_error
            count += x2d.size(0)

    print(f"\nTest MPJPE: {total_error/count:.4f} mm")

if __name__ == "__main__":
    main()
