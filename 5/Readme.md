# 📘 Sparse GCN + Transformer for Real-Time 3D & 2D-in-3D Human Pose Visualization

This repository contains a **hybrid Sparse Graph Convolution + Attention Transformer** framework for lifting 2D keypoints (from MediaPipe) into 3D, plus a collection of utilities to plot “flat” 2D poses in 3D space (Z=0) with bone-level labels. It also includes training scripts, data-processing code, and inference scripts for both static images and (future) video.

---

## ✅ Features

* **Sparse GCN + Transformer Model**
  ‣ Takes normalized 2D (X,Y,0) inputs → predicts 3D (X,Y,Z) joint positions.
  ‣ Residual GCN blocks capture local graph structure; Transformer encodes global joint dependencies.

* **2D-in-3D Flat Plotting**
  ‣ Embed a 2D pose (MediaPipe landmarks) into the plane Z = 0.
  ‣ Flip the Y‐axis so “up” in the image corresponds to “up” in 3D.
  ‣ Draw each bone (MediaPipe’s POSE\_CONNECTIONS) directly on the Z=0 plane.

* **Bone-Level Classification (Upper/Lower/Torso)**
  ‣ Automatically compute “hip line” from MediaPipe’s Left Hip (23) & Right Hip (24).
  ‣ Color bones **red** if both endpoints lie above the hip (upper body),
  ‣ **blue** if both lie below (lower body),
  ‣ **green** if the edge crosses the hip line (torso).

* **Automatic Legend & Joint Labels**
  ‣ Each bone’s midpoint is annotated with “upper”, “lower”, or “torso”.
  ‣ A legend explains the red/blue/green color code.

* **3D Pose Prediction & Alignment**
  ‣ Use `PoseEstimator` (GCN+Transformer) to lift 2D → 3D.
  ‣ Apply SolvePnP to estimate camera extrinsics and align the predicted 3D skeleton with the original image view.

* **Modular Visualization**
  ‣ `visualize.py` draws a standard 2D overlay on an image.
  ‣ `plot_3d_skeleton.py` shows the predicted 3D skeleton (rotated/upright) from a single frame.
  ‣ “Flat” scripts (`plot_2d_in_3d*.py`) show how to embed pure 2D keypoints into a 3D figure.

* **Ease of Use**
  ‣ Pre-saved normalization stats (`pose2d_mean_std.npy`) ensure consistent input scaling.
  ‣ All plotting scripts label joints (0–27) and bones, so you always know which index is which joint.
  ‣ Device-agnostic (CPU/GPU) inference.

---

## 📁 Directory Structure

```
.
├── README.md                        # This file
├── model.py                        # Sparse GCN + Attention Transformer architecture
├── train_pose_model_v2.py         # Training script (MPI-INF-3DHP, MPJPE + bone loss)
├── train_pose_modelV3.py          # (Alternate training version with 3-channel 2D input)
├── extract_mpi_inf_3dhp.py         # Convert raw MPI-INF .mat → compressed .npz
├── mpi_inf_combined.npz           # Preprocessed MPI-INF-3DHP dataset (2D + 3D arrays)
├── pose2d_mean_std.npy            # Saved 2D normalization stats (mean, std for X,Y,0)
├── final_model_weights.pth        # Pretrained GCN+Transformer weights
│
├── predict_image.py               # Run inference on a static image:  
│                                    • MediaPipe 2D → normalize + zero‐pad → GCN+Transformer → 3D  
│                                    • Align and draw skeleton on top of the original image  
│                                    • Saves output as `output.jpg`
│
├── visualize.py                   # 2D overlay utility:  
│                                    • Uses MediaPipe landmarks (33 points) + POSE_CONNECTIONS  
│                                    • Draws red joint circles + green bones directly on BGR frame  
│
├── plot_3d_skeleton.py            # Plot a **predicted 3D skeleton** in upright orientation:  
│                                    • Takes a saved image `input.jpg`  
│                                    • Extracts 2D → normalizes → GCN+Transformer → 3D  
│                                    • Recenters at pelvis, scales to ~1.5 m, and plots joints + bones  
│
├── plot_2d_in_3d_corrected.py      # Embed a **single 2D pose** (MediaPipe) in a 3D plane Z=0:  
│                                    • Detect first 28 landmarks → pixel coords → flip Y  
│                                    • Plot as red dots + green bones (MediaPipe POSE_CONNECTIONS)  
│
├── plot_2d_in_3d_with_bone_labels_and_legend.py  
│                                  # Same “flat Z=0” embedding, but:  
│                                    • Classifies each bone as Upper (red) / Lower (blue) / Torso (green)  
│                                    • Annotates bone midpoint with its label  
│                                    • Adds a legend explaining each color  
│
├── input.jpg                      # Sample input image (replace as needed)
└── output.jpg                     # Last‐saved overlay from `predict_image.py`
```

---

## ⚙️ Setup Instructions

### 1. Create a Conda Environment

```bash
conda create -n sparse_pose_env python=3.10 -y
conda activate sparse_pose_env
```

### 2. Install PyTorch & PyTorch Geometric (with CUDA 12.1)

```bash
pip install torch==2.2.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install torch-geometric
```

### 3. Install Other Dependencies

```bash
pip install mediapipe opencv-python matplotlib einops tqdm imageio
```

> **Note:**
> • If you only plan to run “flat” 2D→3D scripts (Z=0), you do **not** need the PyTorch/PyG installation.
> • For inference with `predict_image.py` or plotting real 3D skeletons (`plot_3d_skeleton.py`), you must have PyTorch and PyTorch Geometric installed.

---

## 🏋️‍♂️ Training Your Own Model

1. Ensure `mpi_inf_combined.npz` is in the root directory. This file contains:

   * `pose2d`: shape `(N_frames, 28, 3)` where each joint is `(x, y, 0)` and has been pre‐saved.
   * `pose3d`: shape `(N_frames, 28, 3)` in millimeters.

2. Run training:

```bash
python train_pose_model_v2.py
```

* **Input:**
  • Each 2D sample already has a dummy Z=0.
  • The dataset loader normalizes using `(pose2d_mean, pose2d_std)` and stores those stats in `pose2d_mean_std.npy`.
* **Architecture:**
  • `PoseEstimator(in_channels=3)` uses two Sparse GCN blocks + a Transformer encoder layer.
  • Final MLP head outputs `(x, y, z)` for each of the 28 joints.
* **Loss:**
  • MPJPE (mean per-joint position error) + 0.01× bone‐length regularization.
* **Output:**
  • Saves best validation weights to `best_model_weights.pth` and final weights to `final_model_weights.pth`.
  • Saves normalization stats to `pose2d_mean_std.npy`.

If you’d prefer 2D input only (shape `(28,2)`), use `train_pose_modelV3.py` and update `PoseEstimator(in_channels=2)` accordingly. That script expects `pose2d` to already be `(28,3)` (with zero Z), so in practice you’d load `(28,2)`, zero-pad it, and then normalize.

---

## 🖼️ Predict & Visualize a 3D Pose from a Single Image

1. Place your test image as `input.jpg` in the root folder (replace the provided sample if needed).

2. Run:

   ```bash
   python predict_image.py
   ```

3. What happens under the hood:

   * **MediaPipe** extracts 33 2D landmarks; we keep only the first 28.
   * We convert normalized `(x,y)` → pixel coords `(px, py)`.
   * Build a `(28,3)` array by zero-padding Z=0.
   * Normalize using `pose2d_mean_std.npy`.
   * Feed into `PoseEstimator(in_channels=3)` → output `(28,3)` in mm.
   * Call SolvePnP → estimate camera rotation + translation to align predicted 3D pose with the original 2D.
   * Draw the 3D skeleton projected back on the original image frame and save as `output.jpg`.

4. Check `output.jpg` to see the overlaid skeleton.

5. If you only want a “flat” 2D-in-3D view (Z=0 plane), skip `predict_image.py` and run either:
   • `plot_2d_in_3d_corrected.py` (basic green-bones overlay)
   • `plot_2d_in_3d_with_bone_labels_and_legend.py` (classified red/blue/green bones + legend).

---

## 🎨 Visualization Utilities

### 1. `visualize.py` (2D Overlay)

```python
from visualize import draw_2d_pose
import cv2, mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(...)

frame = cv2.imread("input.jpeg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(frame)

if results.pose_landmarks:
    output_img = draw_2d_pose(frame.copy(), results.pose_landmarks.landmark)
    cv2.imwrite("output.jpg", output_img)
```

* Draws 33 landmarks (normalized to pixel coords) on the BGR image.
* Bones are green, joints are red, labels are white.

### 2. `plot_3d_skeleton.py` (Upright 3D Skeleton)

```bash
python plot_3d_skeleton.py
```

* Reads `input.jpg`, extracts 28 2D joints, normalizes to `(28,3)`, and predicts a 3D skeleton.
* Recenters at the pelvis (joints 12 & 13), scales the height to \~1.5 m, and plots the upright skeleton.
* X axis = left-right, Y axis = height (Z from raw becomes Y), Z axis = depth (–Y from raw).
* Labels each joint and draws bones as green lines.

### 3. `plot_2d_in_3d_corrected.py` (Flat Z=0 Plot)

```bash
python plot_2d_in_3d_corrected.py
```

* Extracts 28 2D pixel joints → flips Y so “up” in image = “up” in plot.
* Embedded as `(x, h – y, 0)` in 3D → draws green bones using MediaPipe’s `POSE_CONNECTIONS`.
* Ideal for verifying 2D landmark accuracy before adding depth.

### 4. `plot_2d_in_3d_with_bone_labels_and_legend.py` (Flat Z=0 + Bone Classification)

```bash
python plot_2d_in_3d_with_bone_labels_and_legend.py
```

* Same as above but:

  * Computes “hip line” from MediaPipe hips (23 & 24).
  * Colors each bone **red** (upper body), **blue** (lower body), or **green** (torso/mixed).
  * Annotates each bone’s midpoint with its class label.
  * Adds a legend in the upper‐right explaining the color code.

---

## 📦 Module Versions (Tested)

| Library         | Version          |
| --------------- | ---------------- |
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
| imageio         | 2.28.1           |

---

## 🚀 Quick Start Examples

### A. Visualize a “Flat” 2D Pose in 3D

```bash
python plot_2d_in_3d_corrected.py
```

* Input: `input.jpg`
* Output: 3D plot (red joints, green bones on Z=0)

### B. Visualize with Bone Classes + Legend

```bash
python plot_2d_in_3d_with_bone_labels_and_legend.py
```

* Input: `input.jpg`
* Output: 3D plot (red upper bones, blue lower bones, green torso bones + legend)

### C. Predict & Overlay 3D Pose on the Image

```bash
python predict_image.py
```

* Input: `input.jpg`
* Output: `output.jpg` with 3D‑aligned skeleton overlay

### D. Plot Upright 3D Skeleton from Prediction

```bash
python plot_3d_skeleton.py
```

* Input: `input.jpg`
* Output: 3D upright skeleton (centered at pelvis, scaled to \~1.5 m)

---

## ✍️ Citation

If you use this code or model in your research, please cite:

```
Vedansh Tembhre, 2025.
“Hybrid Sparse Graph and Transformer Architecture for Real-Time 3D Human Pose Estimation.”
```

---

## 🔧 Future Enhancements

* **Video/Webcam Support**
  • Real‑time streaming inference (`predict_live.py`).
  • 2D → 3D → flat 3D animation in real time.

* **Advanced Depth Estimation**
  • Integrate a learned 2D→3D lifting (GCN+Transformer) for video frames.
  • Implement temporal smoothing across frames.

* **Bone-Length & Symmetry Regularization**
  • Enforce consistent bone lengths during training/inference for anatomical realism.

* **TorchScript/ONNX Export**
  • For mobile/edge deployment.

* **Camera-Aware Models**
  • Estimate focal length and pose refinement from multi-view or monocular cues.

---

Feel free to open an issue or submit a pull request. Let’s keep making 2D & 3D pose estimation faster, clearer, and more understandable! 🚀
