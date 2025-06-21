import torch
from torch.utils.data import Dataset
import numpy as np

class MPIINF3DHPDataset(Dataset):
    def __init__(self, npz_path, normalize=True):
        data = np.load(npz_path)
        pose2d = data['pose2d'].astype(np.float32)  # (N,28,2) or (N,28,3)
        pose3d = data['pose3d'].astype(np.float32)  # (N,28,3)

        # ensure a confidence channel
        if pose2d.shape[-1] == 2:
            conf = np.ones((pose2d.shape[0], pose2d.shape[1], 1), dtype=np.float32)
            pose2d = np.concatenate([pose2d, conf], axis=-1)

        # normalize per-channel
        flat = pose2d.reshape(-1, 3)
        mean = flat.mean(axis=0)
        std  = flat.std(axis=0) + 1e-6
        if normalize:
            pose2d = (pose2d - mean) / std
            np.save('pose2d_mean_std.npy', np.stack([mean, std]))

        self.pose2d = torch.tensor(pose2d, dtype=torch.float32)  # (N,28,3)
        self.pose3d = torch.tensor(pose3d, dtype=torch.float32)  # (N,28,3)

    def __len__(self):
        return self.pose2d.size(0)

    def __getitem__(self, idx):
        return {
            'pose2d': self.pose2d[idx],   # Tensor (28,3)
            'pose3d': self.pose3d[idx],   # Tensor (28,3)
        }
