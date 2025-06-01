import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from model import PoseEstimator, create_edge_index, device
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Enable for faster training on GPUs
    print("Using GPU for training")
# === Dataset Loader ===
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
            pose2d += np.random.normal(0, 0.015, pose2d.shape)  # stronger augmentation
        return {
            'pose2d': torch.tensor(pose2d),
            'pose3d': torch.tensor(self.pose3d[idx]),
        }

# === Loss Functions ===
def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

def bone_length_loss(preds, edges):
    loss = 0.0
    for i, j in edges:
        bone_lengths = torch.norm(preds[:, i] - preds[:, j], dim=-1)
        loss += torch.mean((bone_lengths - 1.0) ** 2)
    return loss

# === Load Dataset and Dataloaders ===
dataset = MPIINF3DHPDataset("mpi_inf_combined.npz")
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# === Model Setup ===
model = PoseEstimator().to(device)
edge_index = create_edge_index().to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
best_val_loss = float('inf')

# Edge list for bone loss
bone_edges = [
    (0,1), (1,8), (8,12),
    (1,2), (2,3), (3,4),
    (1,5), (5,6), (6,7),
    (8,9), (9,10), (10,11),
    (8,13), (13,14), (14,15),
    (0,16), (0,17)
]

# === Training Loop ===
for epoch in range(200):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/200"):
        inputs = batch['pose2d'].to(device)
        targets = batch['pose3d'].to(device)
        optimizer.zero_grad()
        edge_batched = edge_index.repeat(inputs.size(0), 1, 1).view(2, -1)
        outputs = model(inputs, edge_batched)

        loss_main = mpjpe(outputs, targets)
        loss_reg = bone_length_loss(outputs, bone_edges)
        loss = loss_main + 0.01 * loss_reg

        loss.backward()
        optimizer.step()
        total_train_loss += loss_main.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # === Validation ===
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

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model_weights.pth")
        print("✅ Best model saved to best_model_weights.pth")

torch.save(model.state_dict(), "final_model_weights.pth")
print("✅ Final model saved to final_model_weights.pth")
