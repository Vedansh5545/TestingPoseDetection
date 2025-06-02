# 📘 Sparse GCN + Transformer for Real-Time 3D Human Pose Estimation

This project implements a **hybrid Sparse Graph Convolution + Attention Transformer model** to estimate **3D human poses** from 2D keypoints extracted via MediaPipe. It's lightweight, modular, and optimized for real-time inference with anatomically accurate visualizations and image-aligned predictions.

---

## ✅ Features

- Sparse anatomical graph modeling of joints
- Joint-specific attention routing via Transformers
- Image-aligned 2D → 3D pose normalization and scaling
- Real-time inference on webcam or static images
- Modular `visualize.py` with head/tail toggling and joint coloring
- Device-agnostic (CPU/GPU)

---

## 🧠 Architecture

```
[Image] 
   ↓
[MediaPipe 2D Keypoints] 
   ↓
[Normalization & Zero-padding (x,y → x,y,0)] 
   ↓
[Sparse GCN]
   ↓
[Attention Transformer]
   ↓
[MLP Head → 3D Pose]
   ↓
[Flip + Rotate + Scale Alignment]
   ↓
[draw_3d_pose() → 2D Projection and Display]
```

---

## 📁 Directory Structure

```
.
├── model.py                # GCN + Transformer model
├── train_pose_model_v2.py # Training script with MPJPE loss
├── predict_image.py        # Aligned image-based 3D pose inference
├── visualize.py           # 2D projection from 3D + bone visualization
├── mpi_inf_combined.npz   # Preprocessed MPI-INF-3DHP dataset
├── pose2d_mean_std.npy    # Saved 2D normalization stats
├── model_weights.pth      # Trained model weights
├── input.jpeg             # Sample input image
├── output.jpg             # Visualized output
└── README.md              # This file
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

### 🔹 3. Install PyTorch Geometric

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric
```

### 🔹 4. Install Remaining Dependencies

```bash
pip install mediapipe opencv-python matplotlib einops tqdm
```

---

## 🏋️‍♂️ Training

Ensure your dataset file `mpi_inf_combined.npz` is in the root directory.

```bash
python train_pose_model_v2.py
```

This will:
- Train for 50 epochs using MPJPE loss
- Save weights to `model_weights.pth`
- Save stats to `pose2d_mean_std.npy`

---

## 🖼️ Predict on Image

Place your test image as `input.jpeg`, then run:

```bash
python predict_image.py
```

This will:
- Detect 2D joints with MediaPipe
- Normalize and zero-pad to 3D-compatible input
- Predict 3D joints using GCN + Transformer
- Apply alignment (flip, rotate, scale) to match image orientation
- Draw skeleton on image and save it as `output.jpg`

---

## 🎨 Visualization

`visualize.py` provides:
- Clean 2D joint projection from predicted 3D pose
- Anatomical bone connections (spine, arms, legs, optional head)
- Red joint markers and green lines
- Optional head/face bone toggle (`draw_head=True`)

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
- Webcam/video-based real-time inference (`predict_live.py`)
- TorchScript/ONNX export for deployment
- Bone-length & symmetry regularization in training
- Automatic camera-perspective compensation
- Rotation-invariant keypoint alignment

---

Feel free to open an issue or contribute!  
Let’s make pose estimation faster, cleaner, and smarter 🤖🔥
