import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
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
        return {
            'pose2d': torch.tensor(self.pose2d[idx]),
            'pose3d': torch.tensor(self.pose3d[idx]),
        }

class COCO_MediaPipeDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        pose2d = data['pose2d'].astype(np.float32)

        # Filter based on variance
        filtered = []
        for kp in pose2d:
            if np.std(kp[:, 0]) > 0.02 and np.std(kp[:, 1]) > 0.02:
                filtered.append(kp)

        pose2d = np.array(filtered)
        self.pose2d_mean = pose2d.mean(axis=(0, 1))
        self.pose2d_std = pose2d.std(axis=(0, 1))

        pose2d_norm = (pose2d - self.pose2d_mean) / (self.pose2d_std + 1e-6)
        zeros_z = np.zeros((pose2d_norm.shape[0], 28, 1), dtype=np.float32)

        self.pose2d = np.concatenate([pose2d_norm, zeros_z], axis=-1)
        self.pose3d = self.pose2d

    def __len__(self):
        return len(self.pose2d)

    def __getitem__(self, idx):
        return {
            'pose2d': torch.tensor(self.pose2d[idx]),
            'pose3d': torch.tensor(self.pose3d[idx]),
        }

# Load datasets
dataset1 = MPIINF3DHPDataset("mpi_inf_combined.npz")
dataset2 = COCO_MediaPipeDataset("pose2d_mediapipe_28.npz")

# Combine with weighted sampling
combined_dataset = ConcatDataset([dataset1, dataset2])
weights = [1.0] * len(dataset1) + [0.2] * len(dataset2)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_loader = DataLoader(combined_dataset, batch_size=64, sampler=sampler)

# Initialize model
model = PoseEstimator().to(device)
edge_index = create_edge_index().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        inputs = batch['pose2d'].to(device)
        targets = batch['pose3d'].to(device)

        optimizer.zero_grad()
        edge_batched = edge_index.repeat(inputs.size(0), 1, 1).view(2, -1)
        outputs = model(inputs, edge_batched)

        # Loss without z-penalty
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "model_weights.pth")
print("✅ Model saved to model_weights.pth")