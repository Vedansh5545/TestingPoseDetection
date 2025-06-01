import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from model import PoseEstimator, create_edge_index, device


class MPIINF3DHPDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        pose2d = data['pose2d'].astype(np.float32)
        pose3d = data['pose3d'].astype(np.float32)

        if pose2d.shape[-1] == 2:
            pose2d = np.concatenate([pose2d, np.zeros((pose2d.shape[0], 28, 1), dtype=np.float32)], axis=-1)

        self.pose2d_mean = pose2d.mean(axis=(0, 1))
        self.pose2d_std = pose2d.std(axis=(0, 1))
        self.pose2d = (pose2d - self.pose2d_mean) / (self.pose2d_std + 1e-6)
        self.pose3d = pose3d

        np.save("pose2d_mean_std.npy", np.array([self.pose2d_mean, self.pose2d_std]))

    def __len__(self):
        return len(self.pose2d)

    def __getitem__(self, idx):
        pose2d = self.pose2d[idx].copy()
        if np.random.rand() > 0.5:
            pose2d += np.random.normal(0, 0.01, pose2d.shape)
        return {
            'pose2d': torch.tensor(pose2d),
            'pose3d': torch.tensor(self.pose3d[idx]),
        }


def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))


# === Load dataset and split
dataset = MPIINF3DHPDataset("mpi_inf_combined.npz")
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# === Model, optimizer, scheduler
model = PoseEstimator().to(device)
edge_index = create_edge_index().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float('inf')

# === Training loop
for epoch in range(200):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/200"):
        inputs = batch['pose2d'].to(device)
        targets = batch['pose3d'].to(device)
        optimizer.zero_grad()
        edge_batched = edge_index.repeat(inputs.size(0), 1, 1).view(2, -1)
        outputs = model(inputs, edge_batched)

        if torch.allclose(outputs[0], outputs[1], atol=1e-2):
            print("⚠️ Collapsed output detected.")

        loss = mpjpe(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # === Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['pose2d'].to(device)
            targets = batch['pose3d'].to(device)
            edge_batched = edge_index.repeat(inputs.size(0), 1, 1).view(2, -1)
            outputs = model(inputs, edge_batched)
            val_loss = mpjpe(outputs, targets)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1} | Train MPJPE: {avg_train_loss:.4f} | Val MPJPE: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model_weights.pth")
        print("✅ Best model saved to best_model_weights.pth")

# Save final model
torch.save(model.state_dict(), "final_model_weights.pth")
print("✅ Final model saved to final_model_weights.pth")
