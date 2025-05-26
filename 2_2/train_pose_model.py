import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import PoseEstimator, create_edge_index, device
from tqdm import tqdm

class PoseDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        pose2d = data['pose2d'].astype(np.float32)
        pose3d = data['pose3d'].astype(np.float32)

        self.pose2d_mean = pose2d.mean()
        self.pose2d_std = pose2d.std()

        pose2d = (pose2d - self.pose2d_mean) / (self.pose2d_std + 1e-6)
        self.pose2d = pose2d
        self.pose3d = pose3d

        np.save("pose2d_mean_std.npy", [self.pose2d_mean, self.pose2d_std])

    def __len__(self):
        return len(self.pose2d)

    def __getitem__(self, idx):
        return {
            'pose2d': torch.tensor(self.pose2d[idx]),
            'pose3d': torch.tensor(self.pose3d[idx]),
        }

train_data = PoseDataset("mpi_inf_combined.npz")
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
model = PoseEstimator().to(device)
edge_index = create_edge_index()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        inputs = batch['pose2d'].to(device)
        targets = batch['pose3d'].to(device)

        if torch.isnan(inputs).any() or torch.isnan(targets).any():
            continue

        optimizer.zero_grad()
        outputs = model(inputs, edge_index.repeat(inputs.size(0), 1, 1).view(2, -1))
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "model_weights.pth")
print("✅ Model weights saved to model_weights.pth")