# train_adv.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data_loader import MPIINF3DHPDataset
from model import PoseEstimator, create_edge_index, device
from skeleton_utils import MPIINF_EDGES

# -----------------------------------------------------------------------------
# Loss Helpers
# -----------------------------------------------------------------------------
def mpjpe_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

def bone_length_loss(pred, edge_index):
    edges = set(tuple(e) for e in edge_index.t().cpu().numpy() if e[0] < e[1])
    loss = 0.0
    for i, j in edges:
        loss += torch.mean((pred[:, i] - pred[:, j]).norm(dim=-1))
    return loss / len(edges)

def bone_lengths(pred, edges):
    """
    Compute per-edge bone lengths for a batch.
    Returns tensor of shape (B, E).
    """
    B = pred.size(0)
    lengths = []
    for i, j in edges:
        lengths.append((pred[:, i] - pred[:, j]).norm(dim=-1, keepdim=True))
    return torch.cat(lengths, dim=1)  # (B, E)

# -----------------------------------------------------------------------------
# Adversarial Discriminator
# -----------------------------------------------------------------------------
class BoneDiscriminator(nn.Module):
    def __init__(self, num_edges, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(num_edges)
        self.net = nn.Sequential(
            nn.Linear(num_edges, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, num_edges)
        x = self.norm(x)
        return self.net(x)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_adv(
    num_epochs: int = 50,
    bs: int = 64,
    lr_gen: float = 2e-3,
    lr_disc: float = 1e-4,
    lambda_b: float = 0.01,
    lambda_adv: float = 0.1,
    bone_scale: float = 500.0
):
    # Dataset & loader
    ds = MPIINF3DHPDataset("mpi_inf_combined.npz")
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    # Generator (pretrained) and optimizer
    gen = PoseEstimator().to(device)
    gen.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
    gen_opt = optim.Adam(gen.parameters(), lr=lr_gen)

    # Discriminator and optimizer
    disc = BoneDiscriminator(num_edges=len(MPIINF_EDGES)).to(device)
    disc_opt = optim.Adam(disc.parameters(), lr=lr_disc)

    # Edge index for generator
    edge_index = create_edge_index().to(device)

    bce = nn.BCELoss()

    for epoch in range(1, num_epochs+1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        n_samples = 0

        for batch in loader:
            x2d = batch['pose2d'].to(device)   # (B,28,3)
            y3d = batch['pose3d'].to(device)   # (B,28,3)
            B = x2d.size(0)
            n_samples += B

            # ----------------------------------------
            # 1) Generate fake 3D poses
            # ----------------------------------------
            with torch.no_grad():
                fake3d = gen(x2d, edge_index)  # (B,28,3)

            # ----------------------------------------
            # 2) Prepare real & fake bone-lengths
            # ----------------------------------------
            real_bl = bone_lengths(y3d, MPIINF_EDGES)   / bone_scale  # (B,E)
            fake_bl = bone_lengths(fake3d, MPIINF_EDGES) / bone_scale  # (B,E)

            real_labels = torch.ones((B,1), device=device)
            fake_labels = torch.zeros((B,1), device=device)

            # ----------------------------------------
            # 3) Train Discriminator
            # ----------------------------------------
            disc_opt.zero_grad()
            real_pred = disc(real_bl)
            fake_pred = disc(fake_bl)
            d_loss = 0.5 * (bce(real_pred, real_labels) + bce(fake_pred, fake_labels))
            d_loss.backward()
            disc_opt.step()

            # ----------------------------------------
            # 4) Train Generator
            # ----------------------------------------
            gen_opt.zero_grad()
            pred3d = gen(x2d, edge_index)  # (B,28,3)

            # Standard losses
            mpjpe = mpjpe_loss(pred3d, y3d)
            bone_l = bone_length_loss(pred3d, edge_index)

            # Adversarial loss: want D(fake) -> 1
            bl_norm = bone_lengths(pred3d, MPIINF_EDGES) / bone_scale
            adv_loss = bce(disc(bl_norm), real_labels)

            g_loss = mpjpe + lambda_b * bone_l + lambda_adv * adv_loss
            g_loss.backward()
            gen_opt.step()

            total_d_loss += d_loss.item() * B
            total_g_loss += g_loss.item() * B

        # Epoch summary
        avg_d = total_d_loss / n_samples
        avg_g = total_g_loss / n_samples
        print(f"Epoch {epoch}/{num_epochs} — D_loss: {avg_d:.4f}, G_loss: {avg_g:.4f}")

    # Save improved generator & discriminator
    torch.save(gen.state_dict(), "gen_adv_weights.pth")
    torch.save(disc.state_dict(), "disc_adv_weights.pth")
    print("✅ Adversarial training complete; models saved.")

if __name__ == "__main__":
    train_adv()
