import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from tqdm import tqdm
from model_sdg_rat import SDGRATModel
from torch_geometric.utils import dense_to_sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Dataset Loader with Cleaning and Normalization ===
class MPIINF3DHPDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        pose2d = data['pose2d'].astype(np.float32)
        pose3d = data['pose3d'].astype(np.float32)

        # Add dummy confidence if needed
        if pose2d.shape[-1] == 2:
            conf = np.ones_like(pose2d[..., :1])
            pose2d = np.concatenate([pose2d, conf], axis=-1)

        print(f"📦 Loaded raw data: {pose2d.shape[0]} samples")

        # --- STEP 1: Remove NaNs/Infs
        finite_mask = np.all(np.isfinite(pose2d), axis=(1, 2)) & np.all(np.isfinite(pose3d), axis=(1, 2))

        # --- STEP 2: Remove zero-only samples
        non_zero_mask = np.any(pose2d[..., :2] != 0, axis=(1, 2))

        # --- STEP 3: Remove flat samples (no variance)
        var_mask = np.var(pose2d[..., :2], axis=(1, 2)) > 1e-6

        # --- Final clean mask
        valid_mask = finite_mask & non_zero_mask & var_mask

        pose2d = pose2d[valid_mask]
        pose3d = pose3d[valid_mask]

        print(f"✅ After cleaning: {pose2d.shape[0]} valid samples")

        # --- Normalize
        mean = np.mean(pose2d[..., :2])
        std = np.std(pose2d[..., :2])
        pose2d[..., :2] = (pose2d[..., :2] - mean) / (std + 1e-8)

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
        return torch.tensor(x), torch.tensor(y), self.edge_index


# === MPJPE Metric ===
def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))


# === Bone Length Loss ===
def bone_length_loss(preds, edges):
    loss = 0.0
    for i, j in edges:
        bone_lengths = torch.norm(preds[:, i] - preds[:, j], dim=-1)
        loss += torch.mean((bone_lengths - 1.0) ** 2)
    return loss


# === Skeleton Edges (28-joint MPI-INF-3DHP format) ===
def get_edge_index():
    edges = [
        (0,1), (1,8), (8,12),
        (1,2), (2,3), (3,4),
        (1,5), (5,6), (6,7),
        (8,9), (9,10), (10,11),
        (8,13), (13,14), (14,15),
        (0,16), (0,17)
    ]
    edges += [(j,i) for i,j in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), edges


# === Load Dataset ===
dataset = MPIINF3DHPDataset("mpi_inf_combined.npz")
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# === Model ===
edge_index, bone_edges = get_edge_index()
model = SDGRATModel(edge_count=edge_index.size(1)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# === Training Loop ===
best_val_loss = float('inf')
for epoch in range(1, 101):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x, y, edge = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x, edge_index.to(device))

        if torch.isnan(out).any() or torch.isnan(y).any():
            print("⚠️ NaNs found — skipping this batch")
            continue

        loss_mpjpe = mpjpe(out, y)
        loss_bone = bone_length_loss(out, bone_edges)
        loss = loss_mpjpe + 0.01 * loss_bone
        loss.backward()
        optimizer.step()
        train_loss += loss_mpjpe.item()

    train_loss /= len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y, edge = batch
            x, y = x.to(device), y.to(device)
            out = model(x, edge_index.to(device))

            if torch.isnan(out).any() or torch.isnan(y).any():
                print("⚠️ NaNs found — skipping val batch")
                continue

            val_loss += mpjpe(out, y).item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    print(f"Epoch {epoch} - Train MPJPE: {train_loss:.4f} | Val MPJPE: {val_loss:.4f}")

    if val_loss < best_val_loss and not torch.isnan(torch.tensor(val_loss)):
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_sdg_rat_model.pth")
        print("✅ Saved best model.")

torch.save(model.state_dict(), "final_sdg_rat_model.pth")
print("✅ Training complete. Final model saved.")
