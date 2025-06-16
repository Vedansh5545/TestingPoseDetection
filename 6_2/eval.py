# eval.py

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from data_loader import MPIINF3DHPDataset
from model import PoseEstimator, create_edge_index, device

def mpjpe(pred, gt):
    # pred,gt: [B,J,3] in mm
    return torch.mean(torch.norm(pred - gt, dim=-1)).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mpi_inf_combined.npz",
                        help="Path to .npz with pose2d/pose3d for 28 joints")
    parser.add_argument("--bs",   type=int, default=64,
                        help="Batch size for evaluation")
    args = parser.parse_args()

    # --- Load dataset with SAME normalization as training ---
    ds = MPIINF3DHPDataset(args.data, normalize=True)

    # --- Split off 10% for validation (same split as train.py) ---
    n_val = int(0.1 * len(ds))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val])

    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    # --- Load model ---
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
    model.eval()

    edge_index = create_edge_index().to(device)  # [2, 2*E]

    # --- Evaluate ---
    total_mpjpe = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            x2d = batch["pose2d"].to(device)  # (B,28,3), normalized
            y3d = batch["pose3d"].to(device)  # (B,28,3) in mm
            # replicate edges across batch
            edge_b = edge_index.repeat(1, x2d.size(0)).view(2, -1)
            pred3d = model(x2d, edge_b)       # (B,28,3) in mm
            batch_mpjpe = mpjpe(pred3d, y3d) * x2d.size(0)
            total_mpjpe  += batch_mpjpe
            total_samples += x2d.size(0)

    final_mpjpe = total_mpjpe / total_samples
    print(f"Final Val MPJPE over {total_samples} samples: {final_mpjpe:.3f} mm")


if __name__ == "__main__":
    main()
