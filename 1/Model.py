import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class ModulatedGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ModulatedGraphConv, self).__init__(aggr='mean')  # "Mean" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.mod = torch.nn.Linear(in_channels, out_channels)  # Modulation layer

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Multiply input features by modulation before aggregation
        mod = torch.sigmoid(self.mod(x))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x*mod)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        # Apply a linear transform after aggregation
        return self.lin(aggr_out)

class ModulatedGCN(torch.nn.Module):
    def __init__(self):
        super(ModulatedGCN, self).__init__()
        self.conv1 = ModulatedGraphConv(3, 64)  # Assuming 3 input features (x, y, z)
        self.conv2 = ModulatedGraphConv(64, 128)
        self.conv3 = ModulatedGraphConv(128, 3)  # Output the 3D coordinates

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
