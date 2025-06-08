import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# -------------------
# Learnable Positional Encoding
# -------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_joints, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, dim))

    def forward(self, x):
        return x + self.pos_embed


# -------------------
# Sparse GCN Block with Edge Importance Weighting
# -------------------
class SparseGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_edges):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.num_edges = num_edges
        self.edge_weights = nn.Parameter(torch.ones(num_edges))
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        # Infer batch size from edge_index
        num_total_edges = edge_index.shape[1]
        batch_size = num_total_edges // self.num_edges
        repeated_weights = self.edge_weights.repeat(batch_size).to(x.device)  # [B * E]

        res = self.residual(x)
        x = self.gcn(x, edge_index, edge_weight=repeated_weights)
        x = self.norm(x + res)
        return self.relu(x)


# -------------------
# Routing Attention Transformer Block
# -------------------
class RoutingAttentionTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.routing_bias = nn.Parameter(torch.randn(1, 28, d_model))  # Custom routing bias per joint

    def forward(self, x):
        x = x + self.routing_bias
        x = self.encoder(x)
        return self.norm(x)


# -------------------
# Projection Decoder
# -------------------
class ProjectionDecoder(nn.Module):
    def __init__(self, in_dim, out_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------
# Full SDG-RAT Model
# -------------------
class SDGRATModel(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=128, out_dim=3, num_joints=28, edge_count=34):
        super().__init__()
        self.pos_enc = LearnablePositionalEncoding(num_joints, hidden_dim)
        self.gcn1 = SparseGCNBlock(in_channels, hidden_dim, edge_count)
        self.gcn2 = SparseGCNBlock(hidden_dim, hidden_dim, edge_count)
        self.attn = RoutingAttentionTransformer(d_model=hidden_dim)
        self.decoder = ProjectionDecoder(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        B, J, C = x.shape  # B: batch size, J: joints, C: 2
        x = x.view(-1, C)  # (B*J, 2)
        edge_index = edge_index.repeat(B, 1, 1).view(2, -1)  # Repeat for batch
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = x.view(B, J, -1)
        x = self.pos_enc(x)
        x = self.attn(x)
        return self.decoder(x)
