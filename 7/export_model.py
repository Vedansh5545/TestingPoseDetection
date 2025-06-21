# export_model.py

import torch
from model import PoseEstimator, create_edge_index, device

def main():
    # Instantiate model and load weights
    model = PoseEstimator().to(device)
    model.load_state_dict(torch.load("best_model_weights.pth", map_location=device))
    model.eval()

    # Dummy input: batch of 1, 28 joints, 3 channels
    dummy_2d = torch.randn(1, 28, 3).to(device)
    # Edge index tensor
    edge = create_edge_index().to(device)

    # Trace to TorchScript
    traced = torch.jit.trace(model, (dummy_2d, edge))
    traced.save("pose_estimator.ts")
    print("✅ Saved TorchScript model to pose_estimator.ts")

if __name__ == "__main__":
    main()
