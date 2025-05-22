import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os

# Check for required packages
try:
    from mmpose.models import STGCN
    from mmpose.structures import PoseDataSample
    from mmpose.datasets.transforms import PoseDataPreprocessor
except ImportError:
    raise ImportError(
        "Required packages not found. Install with:\n"
        "pip install mmcv==2.0.0rc4 mmengine==0.10.7 mmpose==1.3.2"
    )

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PoseDataset(Dataset):
    """Custom dataset for 2D-to-3D pose estimation"""
    def __init__(self, npz_path, normalize=True):
        """
        Args:
            npz_path: Path to .npz file containing 'pose2d' and 'pose3d' arrays
            normalize: Whether to normalize 2D inputs
        """
        data = np.load(npz_path)
        self.pose2d = data['pose2d'].astype(np.float32)
        self.pose3d = data['pose3d'].astype(np.float32)
        
        if normalize:
            # Normalize each joint independently
            self.pose2d = (self.pose2d - self.pose2d.mean(axis=(0,1))) / (self.pose2d.std(axis=(0,1)) + 1e-6)

    def __len__(self):
        return len(self.pose2d)

    def __getitem__(self, idx):
        return {
            'pose2d': torch.tensor(self.pose2d[idx]),
            'pose3d': torch.tensor(self.pose3d[idx]),
        }

def create_data_loaders(npz_path, batch_size=64, test_split=0.2):
    """Create train and validation data loaders"""
    full_dataset = PoseDataset(npz_path)
    
    # Split dataset
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - test_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    
    return train_loader, val_loader

def build_model(num_joints=28):
    """Initialize ST-GCN model"""
    return STGCN(
        in_channels=2,  # (x,y) coordinates
        out_channels=3,  # (x,y,z) output
        graph_cfg=dict(
            layout='coco',  # Using COCO layout (change to 'mpi_inf_3dhp' if needed)
            mode='stgcn_spatial'
        ),
        data_preprocessor=dict(type='PoseDataPreprocessor'),
        loss=dict(type='SmoothL1Loss', use_target_weight=True),
    ).cuda()

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """Training loop with validation"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
            for batch in pbar:
                inputs = batch['pose2d'].permute(0, 2, 1).unsqueeze(2).cuda()  # [B,2,1,J]
                targets = batch['pose3d'].cuda()
                
                # Create PoseDataSample objects
                data_samples = [
                    PoseDataSample(gt_instances=dict(keypoints=target.unsqueeze(0))) 
                    for target in targets
                ]
                
                optimizer.zero_grad()
                outputs = model(inputs, data_samples=data_samples)
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]") as pbar:
            for batch in pbar:
                inputs = batch['pose2d'].permute(0, 2, 1).unsqueeze(2).cuda()
                targets = batch['pose3d'].cuda()
                
                data_samples = [
                    PoseDataSample(gt_instances=dict(keypoints=target.unsqueeze(0))) 
                    for target in targets
                ]
                
                outputs = model(inputs, data_samples=data_samples)
                val_loss += outputs['loss'].item()
                pbar.set_postfix({'val_loss': outputs['loss'].item()})
        
        # Logging
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model!")
        
        scheduler.step(avg_val_loss)
    
    return model

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'data_path': "mpi_inf_3dhp.npz",  # Change to your dataset path
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 3e-4,
    }
    
    # Initialize
    train_loader, val_loader = create_data_loaders(
        CONFIG['data_path'], 
        batch_size=CONFIG['batch_size']
    )
    model = build_model()
    
    # Train
    print(f"Starting training with {len(train_loader.dataset)} samples")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate']
    )
    
    print("Training completed! Best model saved to 'best_model.pth'")