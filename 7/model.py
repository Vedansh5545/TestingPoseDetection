# model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        """
        d_model: hidden dimension
        max_len: maximum number of "time steps" (here, joints) to support.
                 Set to >= your joint count (e.g. 28).
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)           # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, J, d_model], J <= max_len
        return x + self.pe[:, : x.size(1), :].to(x.device)


class SparseGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, edge_index):
        # x: [B*J, C], edge_index: [2, E]
        res = self.residual(x)
        x = self.gcn(x, edge_index)
        x = self.norm(x + res)
        return self.relu(x)


class AttentionRoutingTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, max_len=100):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

    def forward(self, x):
        # x: [B, J, d_model]
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.norm(x)


class PoseEstimator(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, out_dim=3, max_joints=28):
        super().__init__()
        self.gcn1 = SparseGCNBlock(in_channels, hidden_dim)
        self.gcn2 = SparseGCNBlock(hidden_dim, hidden_dim)
        # tell the transformer how many joints to expect (<= max_joints)
        self.trans = AttentionRoutingTransformer(
            d_model=hidden_dim, nhead=4, num_layers=2, max_len=max_joints
        )
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index):
        # x: [B, J, C], edge_index: [2, E]
        B, J, C = x.shape
        x = x.view(-1, C)  # [B*J, C]
        # replicate edges for batch dimension:
        edge_index = edge_index.repeat(1, B)
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = x.view(B, J, -1)     # [B, J, hidden_dim]
        x = self.trans(x)        # [B, J, hidden_dim]
        x = self.dropout(x)
        return self.head(x)      # [B, J, out_dim]


def create_edge_index():
    # your original edge list from earlier:
    import torch
    edges = [
        (0, 1), (1, 8), (8, 12),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11),
        (8, 13), (13, 14), (14, 15),
        (0, 16), (0, 17)
    ]
    edges += [(j, i) for i, j in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
