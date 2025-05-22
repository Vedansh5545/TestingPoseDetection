import os
import scipy.io
import numpy as np
from pathlib import Path


def process_mpi_sequence(mat_path, output_path):
    print(f"Processing: {mat_path}")
    mat = scipy.io.loadmat(mat_path)

    # Handle wrapped .mat structure (cell arrays)
    if isinstance(mat['annot2'], np.ndarray) and mat['annot2'].dtype == 'O':
        annot2 = mat['annot2'][0, 0]
        annot3 = mat['annot3'][0, 0]
    else:
        annot2 = mat['annot2']
        annot3 = mat['annot3']

    # Reshape flat vectors into (frames, joints, coords)
    if annot2.ndim == 2 and annot2.shape[1] % 2 == 0:
        num_joints_2d = annot2.shape[1] // 2
        pose2d = annot2.reshape(-1, num_joints_2d, 2)
    else:
        pose2d = np.transpose(annot2, (2, 1, 0))

    if annot3.ndim == 2 and annot3.shape[1] % 3 == 0:
        num_joints_3d = annot3.shape[1] // 3
        pose3d = annot3.reshape(-1, num_joints_3d, 3)
    else:
        pose3d = np.transpose(annot3, (2, 1, 0))

    # Handle missing valid_frame
    total_frames = pose2d.shape[0]
    if 'valid_frame' in mat:
        valid = mat['valid_frame'][0].astype(bool)
    else:
        print("⚠️ No 'valid_frame' found. Assuming all frames are valid.")
        valid = np.ones(total_frames, dtype=bool)

    pose2d = pose2d[valid]
    pose3d = pose3d[valid]

    if pose2d.size == 0 or pose3d.size == 0:
        print("⚠️ No valid frames found. Skipping.")
        return

    np.savez_compressed(output_path, pose2d=pose2d, pose3d=pose3d)
    print(f"✅ Saved to: {output_path}")


def batch_process_all(data_root, output_dir):
    subjects = [f"S{i}" for i in range(1, 9)]
    for subject in subjects:
        for seq_num in range(1, 3):  # Seq1 and Seq2
            mat_path = Path(data_root) / subject / f"Seq{seq_num}" / "annot.mat"
            if not mat_path.exists():
                continue
            output_name = f"{subject}_Seq{seq_num}.npz"
            output_path = Path(output_dir) / output_name
            process_mpi_sequence(mat_path, output_path)


if __name__ == "__main__":
    # ✏️ Set your local dataset and output directories here
    data_root = r"E:\Data1"  # Folder containing S1 to S8
    output_dir = r"C:\Users\vedan\Downloads\2posedetectionimplementation\npz"

    os.makedirs(output_dir, exist_ok=True)
    batch_process_all(data_root, output_dir)
    pose2d_all = []
    pose3d_all = []

    for file in sorted(Path(r"C:\Users\vedan\Downloads\2posedetectionimplementation\npz").glob("*.npz")):
        data = np.load(file)
        pose2d_all.append(data['pose2d'])
        pose3d_all.append(data['pose3d'])

    pose2d = np.concatenate(pose2d_all)
    pose3d = np.concatenate(pose3d_all)

    np.savez_compressed("mpi_inf_combined.npz", pose2d=pose2d, pose3d=pose3d)
