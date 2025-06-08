import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch_geometric.utils import dense_to_sparse
import numpy as np
from tqdm import tqdm
from model_sdg_rat import SDGRATModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Dataset Loader ===
class MPIINF3DHPDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        pose2d = data['pose2d'].astype(np.float32)
        pose3d = data['pose3d'].astype(np.float32)

        # Sanitize NaNs or Infs
        pose2d = np.nan_to_num(pose2d)
        pose3d = np.nan_to_num(pose3d)

        if pose2d.shape[-1] == 2:
            conf = np.ones_like(pose2d[..., :1])  # (N, 28, 1)
            pose2d = np.concatenate([pose2d, conf], axis=-1)  # (N, 28, 3)

        self.pose2d = pose2d
        self.pose3d = pose3d
        self.edge_index = self.build_edge_index(pose2d.shape[1])

    def build_edge_index(self, num_nodes):
        adj = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        edge_index = dense_to_sparse(torch.tensor(adj, dtype=torch.float))[0]
        return edge_index

    def __len__(self):
        return len(self.pose2d)

    def __getitem__(self, idx):
        x = self.pose2d[idx, :, :2]  # x, y
        y = self.pose3d[idx]
        return x, y, self.edge_index


# === Loss Functions ===
def mpjpe(pred, target):
    return torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, dim=-1) + 1e-6))

def bone_length_loss(preds, edges):
    loss = 0.0
    for i, j in edges:
        bone_lengths = torch.norm(preds[:, i] - preds[:, j], dim=-1)
        loss += torch.mean((bone_lengths - 1.0) ** 2)
    return loss


# === Skeleton Edges (MPI-INF-3DHP style) ===
def get_edge_index():
    edges = [
        (0,1), (1,8), (8,12),
        (1,2), (2,3), (3,4),
        (1,5), (5,6), (6,7),
        (8,9), (9,10), (10,11),
        (8,13), (13,14), (14,15),
        (0,16), (0,17)
    ]
    edges += [(j, i) for i, j in edges]  # Bidirectional
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), edges


# === Load Dataset ===
dataset = MPIINF3DHPDataset("mpi_inf_combined.npz")
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# === Model Setup ===
edge_index, bone_edges = get_edge_index()
model = SDGRATModel(edge_count=edge_index.size(1)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float('inf')
for epoch in range(1, 101):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x, y, batch_edge_index = batch
        x = x.to(device)
        y = y.to(device)
        edge_idx = edge_index.to(device)

        optimizer.zero_grad()
        out = model(x, edge_idx)

        loss_mpjpe = mpjpe(out, y)
        loss_bone = bone_length_loss(out, bone_edges)
        loss = loss_mpjpe + 0.01 * loss_bone

        if torch.isnan(loss):
            print("⚠️ NaN detected in loss!")
            print("x:", x)
            print("y:", y)
            print("out:", out)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss_mpjpe.item()

    train_loss /= len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            out = model(x, edge_index.to(device))
            val_loss += mpjpe(out, y).item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch} - Train MPJPE: {train_loss:.4f} | Val MPJPE: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_sdg_rat_model.pth")
        print("✅ Saved best model.")

torch.save(model.state_dict(), "final_sdg_rat_model.pth")
print("✅ Training complete. Final model saved.")
