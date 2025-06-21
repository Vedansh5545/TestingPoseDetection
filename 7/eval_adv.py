# eval_adv.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import MPIINF3DHPDataset
from model import PoseEstimator, create_edge_index, device

def mpjpe(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))

def evaluate(
    weight_path: str = "gen_adv_weights.pth",
    data_path:   str = "mpi_inf_combined.npz",
    batch_size:  int = 64
):
    # 1) Load dataset & split off 10% for val
    ds = MPIINF3DHPDataset(data_path)
    n_val = int(0.1 * len(ds))
    _, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 2) Load model
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    edge_index = create_edge_index().to(device)

    # 3) Compute MPJPE on val set
    total_err = 0.0
    total_joints = 0
    with torch.no_grad():
        for batch in val_loader:
            x2d = batch['pose2d'].to(device)   # (B,28,3)
            y3d = batch['pose3d'].to(device)   # (B,28,3)
            pred3d = model(x2d, edge_index)    # (B,28,3)

            # sum over batch and joints
            errs = torch.norm(pred3d - y3d, dim=-1)  # (B,28)
            total_err += errs.sum().item()
            total_joints += errs.numel()

    mean_mpjpe = total_err / total_joints  # in same units as your data (mm)
    print(f"Final Val MPJPE (adv): {mean_mpjpe:.2f} mm")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate adversarial generator")
    p.add_argument("--weights", default="gen_adv_weights.pth",
                   help="Path to adversarially‐trained generator weights")
    p.add_argument("--data", default="mpi_inf_combined.npz",
                   help="Path to preprocessed MPI-INF-3DHP .npz")
    p.add_argument("--bs", type=int, default=64, help="Val batch size")
    args = p.parse_args()

    evaluate(args.weights, args.data, args.bs)
