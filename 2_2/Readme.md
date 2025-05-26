# 📘 Sparse GCN + Transformer for Real-Time 3D Human Pose Estimation

This project implements a **hybrid Sparse Graph Convolution + Attention Transformer model** to estimate **3D human poses** from 2D keypoints extracted via MediaPipe. It's lightweight, modular, and optimized for real-time inference.

---

## ✅ Features
- Sparse edge graph modeling of joints
- Joint-specific attention routing via Transformers
- End-to-end 2D to 3D pipeline
- Real-time inference on webcam or static images
- Device-agnostic (CPU/GPU)

---

## 🧠 Architecture
```
[Image] --> [MediaPipe 2D Keypoints] --> [Sparse GCN] --> [Transformer] --> [MLP Head] --> [3D Pose]
```

---

## 📁 Directory Structure
```
.
├── model.py                # Model architecture (GCN + Transformer)
├── train_pose_model.py     # Training script
├── predict_image_3dplot.py # Inference on single image
├── mpi_inf_combined.npz    # Preprocessed MPI-INF-3DHP dataset
├── pose2d_mean_std.npy     # Saved 2D normalization stats
├── input.jpeg              # Test input image
├── output_3d.png           # Saved output plot
└── README.md               # This file
```

---

## ⚙️ Setup Instructions

### 🔹 1. Create Conda Environment
```bash
conda create -n sparse_pose_env python=3.10 -y
conda activate sparse_pose_env
```

### 🔹 2. Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.1.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 🔹 3. Install PyTorch Geometric + Dependencies
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric
```

### 🔹 4. Install Remaining Requirements
```bash
pip install mediapipe opencv-python matplotlib einops tqdm
pip install mmcv==2.0.0rc4 mmengine==0.10.7 mmpose==1.3.2
```

---

## 🏋️‍♂️ Training

Ensure your dataset file `mpi_inf_combined.npz` is in the root directory.

```bash
python train_pose_model.py
```
This will:
- Train for 10 epochs
- Save model weights to `model_weights.pth`
- Save normalization stats to `pose2d_mean_std.npy`

---

## 🖼️ Predict on Image

Place your input image as `input.jpeg` (or change the filename in script) and run:

```bash
python predict_image_3dplot.py
```
This will:
- Run MediaPipe to extract 2D keypoints
- Normalize using training dataset mean/std
- Predict 3D pose
- Plot and save the result as `output_3d.png`

---

## 📦 Module Versions
| Library         | Version         |
|----------------|------------------|
| Python          | 3.10.x           |
| torch           | 2.1.0+cu121      |
| torchvision     | 0.16.0+cu121     |
| torch-geometric | 2.4.0            |
| torch-scatter   | 2.1.2+pt21cu121  |
| torch-sparse    | 0.6.18+pt21cu121 |
| mediapipe       | 0.10.9           |
| opencv-python   | 4.9.0.80         |
| matplotlib      | 3.8.4            |
| einops          | 0.7.0            |
| tqdm            | 4.66.2           |

---

## ✍️ Citation
```
Vedansh Tembhre, 2025. "Hybrid Sparse Graph and Transformer Architecture for Real-Time 3D Human Pose Estimation."
```

---

## 🔧 Future Enhancements
- MPJPE & P-MPJPE evaluation metrics
- Webcam-based real-time inference
- Export to ONNX or TensorRT
- Fine-tuned graph structures based on dataset type

---

Feel free to open an issue or contribute!
