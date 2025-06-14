# train.py

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data_loader import MPIINF3DHPDataset
from model import PoseEstimator, create_edge_index, device

def mpjpe(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))

def bone_length_loss(preds, edge_index):
    # edge_index: [2, 2*E] bidirectional
    edges = set(tuple(e) for e in edge_index.t().cpu().numpy() if e[0] < e[1])
    loss = 0.0
    for i, j in edges:
        loss += torch.mean((preds[:, i] - preds[:, j]).norm(dim=-1))
    return loss / len(edges)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="mpi_inf_combined.npz",
                   help="Path to .npz with pose2d/pose3d")
    p.add_argument("--epochs", type=int, default=30,
                   help="Number of training epochs")
    p.add_argument("--bs",     type=int, default=64,
                   help="Batch size")
    p.add_argument("--lr",     type=float, default=1e-3,
                   help="Learning rate for Adam optimizer")
    args = p.parse_args()

    # ─── Dataset ──────────────────────────────────────────────────────────
    ds = MPIINF3DHPDataset(args.data)
    n_train = int(0.9 * len(ds))
    n_val   = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs)

    # ─── Model & Optimizer ───────────────────────────────────────────────
    model      = PoseEstimator().to(device)
    edge_index = create_edge_index().to(device)            # [2,2E]
    optimizer  = optim.Adam(model.parameters(), lr=args.lr) 

    best_loss = float('inf')

    # ─── Training Loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train = 0.0

        for batch in train_loader:
            x2d = batch['pose2d'].to(device)   # (B, J, 3)
            y3d = batch['pose3d'].to(device)   # (B, J, 3)

            optimizer.zero_grad()
            # replicate edge_index for batch dimension
            edge_b = edge_index.repeat(1, x2d.size(0)).view(2, -1)
            pred3d = model(x2d, edge_b)

            loss = mpjpe(pred3d, y3d) + 0.01 * bone_length_loss(pred3d, edge_index)
            loss.backward()
            optimizer.step()

            total_train += loss.item() * x2d.size(0)

        avg_train = total_train / len(ds)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {avg_train:.4f}")

        # (Optional) Validation
        model.eval()
        with torch.no_grad():
            total_val = 0.0
            for batch in val_loader:
                x2d = batch['pose2d'].to(device)
                y3d = batch['pose3d'].to(device)
                edge_b = edge_index.repeat(1, x2d.size(0)).view(2, -1)
                pred3d = model(x2d, edge_b)
                total_val += mpjpe(pred3d, y3d).item() * x2d.size(0)
            avg_val = total_val / len(val_ds)
        print(f"           Val MPJPE: {avg_val:.4f}")

        # save best
        if avg_train < best_loss:
            best_loss = avg_train
            torch.save(model.state_dict(), "best_model_weights.pth")
            print("✅ Saved best_model_weights.pth")

if __name__ == "__main__":
    main()
