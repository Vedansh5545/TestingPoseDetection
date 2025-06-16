# bone_label_classifier.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skeleton_utils import MPIINF_EDGES            # your 28-joint edge list :contentReference[oaicite:4]{index=4}
from overlay_2d_labeled import BONE_LABELS         # semantic names for major bones :contentReference[oaicite:5]{index=5}

# 1. Build bone‐name ↔ class‐id mappings
bone_names = sorted(set(BONE_LABELS.values()))
bone_to_id = {name: idx for idx, name in enumerate(bone_names)}

class BoneLabelDataset(Dataset):
    """
    Produces one sample per (frame, bone) pair:
      - feats:  [xi, yi, xj, yj, dx, dy, dist, ci, cj] (9-dim)
      - label:  integer in [0, num_classes) or -1 for “ignore”
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pose2d = data['pose2d']      # shape: (N_frames, 28, 3)
        self.edges  = MPIINF_EDGES
        self.bone_to_id = bone_to_id

    def __len__(self):
        return len(self.pose2d) * len(self.edges)

    def __getitem__(self, idx):
        n_edges = len(self.edges)
        frame_idx = idx // n_edges
        edge_idx  = idx %  n_edges
        i, j = self.edges[edge_idx]

        p = self.pose2d[frame_idx]        # (28,3)
        xi, yi, ci = p[i]
        xj, yj, cj = p[j]
        dx, dy = xj - xi, yj - yi
        dist = np.hypot(dx, dy)

        feats = np.array([xi, yi, xj, yj, dx, dy, dist, ci, cj], dtype=np.float32)
        name  = BONE_LABELS.get((i, j), None)

        # label = -1 for bones we’re not semantically naming
        label = self.bone_to_id[name] if name is not None else -1
        return torch.from_numpy(feats), label

class BoneClassifier(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, num_classes=len(bone_names)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_classifier(npz_path, epochs=5, batch_size=512, lr=1e-3):
    ds = BoneLabelDataset(npz_path)
    # Filter out samples with label = -1
    good_idxs = [i for i in range(len(ds)) if ds[i][1] >= 0]
    sub = torch.utils.data.Subset(ds, good_idxs)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=True, drop_last=True)

    model = BoneClassifier().to('cpu')
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        total_loss = 0.0
        total_acc  = 0
        N = 0
        model.train()
        for feats, labs in loader:
            logits = model(feats)
            loss   = lossf(logits, labs)
            opt.zero_grad(); loss.backward(); opt.step()

            total_loss += loss.item() * feats.size(0)
            total_acc  += (logits.argmax(dim=1)==labs).sum().item()
            N += feats.size(0)

        print(f"Epoch {ep}/{epochs} — Loss: {total_loss/N:.4f}, "
              f"Accuracy: {total_acc/N:.3f}")

    torch.save(model.state_dict(), "bone_classifier.pth")
    print("✅ Saved bone_classifier.pth")
    return model

if __name__ == "__main__":
    # Train on your MPI-INF data
    train_classifier("mpi_inf_combined.npz", epochs=10, batch_size=512, lr=1e-3)
