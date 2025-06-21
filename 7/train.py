# train.py

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data_loader import MPIINF3DHPDataset
from model import PoseEstimator, create_edge_index, device

def mpjpe(pred, gt):
    """Mean per‐joint position error."""
    return torch.mean(torch.norm(pred - gt, dim=-1))

def bone_length_loss(preds, edge_index):
    """
    Bone‐length regularizer: average L2 distance between connected joints.
    """
    # edge_index: (2, E)
    E = edge_index.size(1)
    loss = 0.0
    for e in range(E):
        i, j = edge_index[0, e].item(), edge_index[1, e].item()
        loss += torch.mean((preds[:, i, :] - preds[:, j, :]).norm(dim=-1))
    return loss / E

def main():
    parser = argparse.ArgumentParser(
        description="Train Sparse‐GCN+Transformer pose lifter with bone & depth losses"
    )
    parser.add_argument(
        "--data", "-d",
        default="mpi_inf_combined.npz",
        help="Path to .npz containing 'pose2d' and 'pose3d'"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--bs", "--batch_size",
        type=int, default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int, default=10,
        help="Early‐stopping patience (epochs without val improvement)"
    )
    parser.add_argument(
        "--bone_weight",
        type=float, default=0.01,
        help="Weight for bone‐length loss term"
    )
    parser.add_argument(
        "--depth_weight",
        type=float, default=1.0,
        help="Weight for pure Z‐axis MSE loss (try 1.0→5.0)"
    )
    args = parser.parse_args()

    # 1) Load & split dataset
    ds = MPIINF3DHPDataset(args.data, normalize=True)
    n_train = int(0.9 * len(ds))
    n_val   = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False)

    # 2) Model, optimizer, scheduler
    model      = PoseEstimator().to(device)
    edge_index = create_edge_index().to(device)
    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val = float('inf')
    patience_counter = 0

    # 3) Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train = 0.0

        for batch in train_loader:
            x2d = batch['pose2d'].to(device)   # (B,28,3)
            y3d = batch['pose3d'].to(device)   # (B,28,3)

            optimizer.zero_grad()
            pred3d = model(x2d, edge_index)    # (B,28,3)

            # MPJPE loss
            loss_mpjpe = mpjpe(pred3d, y3d)

            # Bone‐length consistency loss
            loss_bone  = bone_length_loss(pred3d, edge_index)

            # Depth‐only MSE loss
            z_gt   = y3d   [:,:,2]   / 1000.0
            z_pred = pred3d[:,:,2] / 1000.0  # mm → m
            loss_depth = F.mse_loss(z_pred, z_gt)

            # Combine losses
            loss = (
                loss_mpjpe
              + args.bone_weight  * loss_bone
              + args.depth_weight * loss_depth
            )

            loss.backward()
            optimizer.step()
            total_train += loss.item() * x2d.size(0)
            
        print(f"MPJPE={loss_mpjpe:.3f}  Bone={loss_bone:.3f}  Depth={loss_depth:.3f}")
        avg_train = total_train / len(train_ds)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {avg_train:.4f}")

        # 4) Validation MPJPE only
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x2d = batch['pose2d'].to(device)
                y3d = batch['pose3d'].to(device)
                pred3d = model(x2d, edge_index)
                total_val += mpjpe(pred3d, y3d).item() * x2d.size(0)

        avg_val = total_val / len(val_ds)
        print(f"           Val MPJPE: {avg_val:.4f}")

        # 5) Scheduler & early stopping
        scheduler.step(avg_val)
        if avg_val < best_val - 1e-4:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_weights.pth")
            print("✅ Saved best_model_weights.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"⏱ Early stopping after {epoch} epochs without improvement")
                break

    print("Training complete.")

if __name__ == "__main__":
    main()
