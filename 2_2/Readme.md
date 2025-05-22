# 📘 3D Human Pose Estimation with Sparse GCN + Attention Routing

## 🧠 Project Overview
This project demonstrates a real-time **3D human pose detection** system using a **hybrid Sparse Graph Convolution + Transformer model**. It is optimized for:
- ✅ Real-time speed
- ✅ High accuracy
- ✅ Novel attention-based architecture for routing joint relevance dynamically

> 🏆 Inspired by cutting-edge research like MotionAGFormer and PoseFormer, but introducing **sparse dynamic graph learning and joint-specific attention**.

---

## 📐 Architecture
```
Input: Webcam Frame → 2D Keypoints (MediaPipe)
   ↓
Sparse Graph Convolution (learns dynamic joint connectivity)
   ↓
Attention Routing Transformer (focuses only on relevant joints)
   ↓
MLP → 3D Joint Position Estimation
   ↓
Real-time 3D Skeleton Overlay (on camera frame)
```

---

## 📦 Components
- `model.py`: Defines `SparseGCNLayer`, `AttentionRoutingTransformer`, and `PoseEstimator`
- `predict_live.py`: Real-time webcam inference with MediaPipe keypoints
- `visualize.py`: Projects and renders 3D pose on the frame
- `train_pose_model.py`: Trains the model on `mpi_inf_combined.npz`
- `extract_mpi_inf_3dhp.py`: Converts `.mat` MPI-INF-3DHP data to `.npz` format

---

## 🖥️ Live Demo (Run This)
```bash
python predict_live.py
```
- Requires a webcam
- Press `Esc` to exit

---

## 🏋️‍♂️ Training
Train your own model using:
```bash
python train_pose_model.py
```
Ensure `mpi_inf_combined.npz` is present in the working directory.

---

## 📂 Dataset
- Source: [MPI-INF-3DHP](http://vcai.mpi-inf.mpg.de/3dhp-dataset/)
- Converted to NumPy `.npz` using `extract_mpi_inf_3dhp.py`
- Format:
  - `pose2d`: (N, 28, 2)
  - `pose3d`: (N, 28, 3)

---

## ⚙️ Setup Instructions

### 🧪 Requirements
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install mediapipe opencv-python matplotlib einops tqdm
```

### 🧠 GPU
Ensure your system has **CUDA 12.8** and an NVIDIA GPU enabled.

---

## 📊 Evaluation (Coming Soon)
Add MPJPE, P-MPJPE, and acceleration metrics to validate accuracy.

---

## 🔬 Research Highlights
- 🔄 Uses sparse edge index (not full graphs)
- 🧠 Each joint attends only to relevant joints (routing attention)
- 🪶 Lightweight GCN + Transformer stack, real-time ready
- 🎯 Highly modular and explainable design

---

## ✍️ Citation (if used)
```
Vedansh Tembhre, 2025. "Hybrid Sparse Graph and Transformer Architecture for Real-Time 3D Human Pose Estimation."
```
