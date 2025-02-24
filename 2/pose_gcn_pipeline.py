#!/usr/bin/env python
"""
A complete pipeline for 3D human pose estimation using a Modulated Graph Convolutional Network (GCN)
on the MPI-INF-3DHP dataset.

This script performs:
    - Data loading and cleaning (from a .mat annotation file)
    - Model definition (modulated GCN layers)
    - Training and testing of the model

Usage:
    python pose_gcn_pipeline.py --annot_path path/to/annot.mat --num_epochs 50
"""

import os
import math
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ========================
# Model Definition Section
# ========================

class ModulatedGraphConv(nn.Module):
    """
    A modulated graph convolution layer.
    This layer performs a standard graph convolution with an additional learnable modulation matrix.
    
    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        num_joints (int): Number of nodes (joints) in the graph.
        adjacency_matrix (torch.Tensor): Fixed skeleton adjacency matrix (shape: [num_joints, num_joints]).
        bias (bool): Whether to include a bias term.
    """
    def __init__(self, in_channels, out_channels, num_joints, adjacency_matrix, bias=True):
        super(ModulatedGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints

        # Learnable modulation matrix (initialized to ones)
        self.modulation = nn.Parameter(torch.ones(num_joints, num_joints))
        # Weight matrix for transforming input features
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        # Fixed adjacency matrix
        self.register_buffer('A', adjacency_matrix)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the weight matrix using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # x: [batch_size, num_joints, in_channels]
        x_transformed = torch.matmul(x, self.weight)  # [batch, num_joints, out_channels]
        # Element-wise modulate the fixed adjacency matrix
        A_modulated = self.A * self.modulation  # [num_joints, num_joints]
        # Graph convolution: aggregate features from neighbors
        out = torch.matmul(A_modulated, x_transformed)
        if self.bias is not None:
            out = out + self.bias
        return out


class PoseGCN(nn.Module):
    """
    A stacked modulated GCN for 3D human pose estimation.
    
    Args:
        in_channels (int): Input channel dimension (e.g., 2 for 2D keypoints).
        hidden_channels (int): Hidden channel dimension.
        out_channels (int): Output channel dimension (e.g., 3 for 3D coordinates).
        num_joints (int): Number of joints in the skeleton.
        adjacency_matrix (torch.Tensor): Fixed skeleton adjacency matrix.
        num_layers (int): Total number of GCN layers.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_joints, adjacency_matrix, num_layers=3):
        super(PoseGCN, self).__init__()
        layers = []
        # First layer: from input to hidden representation
        layers.append(ModulatedGraphConv(in_channels, hidden_channels, num_joints, adjacency_matrix))
        # Intermediate layers: hidden to hidden
        for _ in range(num_layers - 2):
            layers.append(ModulatedGraphConv(hidden_channels, hidden_channels, num_joints, adjacency_matrix))
        # Final layer: from hidden to output (3D coordinates)
        layers.append(ModulatedGraphConv(hidden_channels, out_channels, num_joints, adjacency_matrix))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


# ============================
# Data Loading & Cleaning
# ============================

class MPIINF3DHPCDataset(Dataset):
    def __init__(self, annot_path, num_joints=17, transform=None):
        self.num_joints = num_joints
        data = sio.loadmat(annot_path)

        # Get the 3D annotations: prefer 'univ_annot3', fallback to 'annot3'
        if 'univ_annot3' in data:
            annot3 = data['univ_annot3']
        elif 'annot3' in data:
            annot3 = data['annot3']
        else:
            raise KeyError("Neither 'univ_annot3' nor 'annot3' found in the .mat file.")

        if 'annot2' not in data:
            raise KeyError("'annot2' not found in the .mat file.")
        annot2 = data['annot2']

        # A helper function to process the annotation.
        # This function supports two common storage schemes:
        # 1. Cell-based: where annot is a cell array of length (num_joints*dims),
        #    each cell containing a vector of length (num_frames).
        # 2. Frame-based: where each element corresponds to one frame’s vector.
        def process_annotation(annot, num_joints, dims):
            annot = np.array(annot)
            flat_annot = annot.flatten()
            # If the number of cells equals num_joints*dims, assume cell-based.
            if flat_annot.size == num_joints * dims:
                cell_arrays = [np.asarray(x).flatten() for x in flat_annot]
                num_frames = cell_arrays[0].shape[0]
                for i, arr in enumerate(cell_arrays):
                    if arr.shape[0] != num_frames:
                        raise ValueError(f"Cell {i} has {arr.shape[0]} frames; expected {num_frames}.")
                # Stack so that each row corresponds to a frame and columns to coordinates.
                combined = np.stack(cell_arrays, axis=1)  # shape: (num_frames, num_joints*dims)
                return combined
            else:
                # Otherwise, assume each element in flat_annot corresponds to one frame.
                processed = []
                for idx, x in enumerate(flat_annot):
                    arr = np.asarray(x).flatten()
                    if arr.size % (num_joints * dims) == 0 and arr.size != num_joints * dims:
                        num_frames_in_cell = arr.size // (num_joints * dims)
                        arr = arr.reshape(num_frames_in_cell, num_joints * dims)
                        for frame in arr:
                            processed.append(frame.astype(np.float32))
                    else:
                        if arr.size != num_joints * dims:
                            raise ValueError(f"Frame {idx} has {arr.size} elements, expected {num_joints * dims}.")
                        processed.append(arr.astype(np.float32))
                return np.stack(processed, axis=0)

        # Process the 2D and 3D annotations.
        kp2d_array = process_annotation(annot2, num_joints, dims=2)
        kp3d_array = process_annotation(annot3, num_joints, dims=3)

        print("Processed annot2 shape:", kp2d_array.shape)
        print("Processed annot3 shape:", kp3d_array.shape)

        # Reshape arrays into the desired format:
        # 2D keypoints: (num_frames, num_joints, 2)
        # 3D keypoints: (num_frames, num_joints, 3)
        self.kp2d = kp2d_array.reshape(-1, num_joints, 2)
        self.kp3d = kp3d_array.reshape(-1, num_joints, 3)
        self.transform = transform

    def __len__(self):
        return self.kp2d.shape[0]
    
    def __getitem__(self, idx):
        # Ensure the tensors are created with float32 dtype.
        kp2d = torch.tensor(self.kp2d[idx], dtype=torch.float32)
        kp3d = torch.tensor(self.kp3d[idx], dtype=torch.float32)
        sample = {'kp2d': kp2d, 'kp3d': kp3d}
        if self.transform:
            sample = self.transform(sample)
        return sample


# ============================
# Training and Testing Routines
# ============================

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        kp2d = batch['kp2d'].to(device)
        kp3d = batch['kp3d'].to(device)
        optimizer.zero_grad()
        outputs = model(kp2d)
        loss = criterion(outputs, kp3d)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * kp2d.size(0)
    return running_loss / len(loader.dataset)

def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            kp2d = batch['kp2d'].to(device)
            kp3d = batch['kp3d'].to(device)
            outputs = model(kp2d)
            loss = criterion(outputs, kp3d)
            running_loss += loss.item() * kp2d.size(0)
    return running_loss / len(loader.dataset)


# ============================
# Main Function
# ============================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load and split the dataset (80% train, 20% test)
    dataset = MPIINF3DHPCDataset(args.annot_path, num_joints=args.num_joints)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Define a simple fixed adjacency matrix.
    # (For real applications, use the actual skeleton connectivity.)
    A = torch.eye(args.num_joints)
    
    # Create the PoseGCN model
    model = PoseGCN(in_channels=2, hidden_channels=args.hidden_channels, out_channels=3,
                    num_joints=args.num_joints, adjacency_matrix=A, num_layers=args.num_layers).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        if test_loss < best_loss:
            best_loss = test_loss
            # Save the best model
            torch.save(model.state_dict(), args.model_path)
    
    print("Training completed. Best test loss:", best_loss)


# ============================
# Entry Point
# ============================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modulated GCN for 3D Human Pose Estimation on MPI-INF-3DHP")
    parser.add_argument('--annot_path', type=str, default='path/to/annot.mat', help="Path to annotation .mat file")
    parser.add_argument('--num_joints', type=int, default=17, help="Number of joints in the skeleton")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--hidden_channels', type=int, default=64, help="Number of hidden channels in GCN")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of GCN layers")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--model_path', type=str, default='best_pose_gcn.pth', help="Path to save the best model")
    
    args = parser.parse_args()
    main(args)
