import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import PoseEstimator
from tqdm import tqdm

class PoseDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        pose2d = data['pose2d'].astype(np.float32)
        pose3d = data['pose3d'].astype(np.float32)

        # Normalize 2D inputs
        pose2d = (pose2d - pose2d.mean()) / (pose2d.std() + 1e-6)
        self.pose2d = pose2d
        self.pose3d = pose3d

    def __len__(self):
        return len(self.pose2d)

    def __getitem__(self, idx):
        return {
            'pose2d': torch.tensor(self.pose2d[idx]),
            'pose3d': torch.tensor(self.pose3d[idx]),
        }

train_data = PoseDataset("mpi_inf_combined.npz")
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

def create_edge_index(num_joints=28):
    edges = []
    for i in range(num_joints - 1):
        edges.append([i, i+1])
    return torch.tensor(edges + [list(reversed(e)) for e in edges], dtype=torch.long).t().contiguous()

edge_index = create_edge_index().cuda()
model = PoseEstimator().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        inputs = batch['pose2d'].cuda()
        targets = batch['pose3d'].cuda()

        if torch.isnan(inputs).any() or torch.isnan(targets).any():
            continue

        optimizer.zero_grad()
        outputs = model(inputs, edge_index.repeat(inputs.size(0), 1, 1).view(2, -1))
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model_weights.pth")
print("✅ Model weights saved to model_weights.pth")
