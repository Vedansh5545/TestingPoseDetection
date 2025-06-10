import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Step 1: Approximate 2D Joint Coordinates from output.jpg ===
# These are mock pixel values approximated from your image
joints_2d = np.array([
    [250, 100],  # 0: Head
    [250, 140],  # 1: Neck
    [200, 140],  # 2: Left Shoulder
    [170, 190],  # 3: Left Elbow
    [160, 240],  # 4: Left Wrist
    [300, 140],  # 5: Right Shoulder
    [330, 180],  # 6: Right Elbow
    [340, 230],  # 7: Right Wrist
    [250, 190],  # 8: Spine
    [250, 240],  # 9: Hip
    [220, 320],  # 10: Left Knee
    [215, 400],  # 11: Left Ankle
    [275, 320],  # 12: Right Knee
    [280, 400]   # 13: Right Ankle
])

# === Step 2: Skeleton edges and biological bone lengths (in cm) ===
edges = [
    (0,1), (1,2), (2,3), (3,4),
    (1,5), (5,6), (6,7),
    (1,8), (8,9),
    (9,10), (10,11),
    (9,12), (12,13)
]

bone_lengths_cm = {
    (0,1): 25, (1,2): 18, (2,3): 30, (3,4): 25,
    (1,5): 18, (5,6): 30, (6,7): 25,
    (1,8): 30, (8,9): 25,
    (9,10): 40, (10,11): 40,
    (9,12): 40, (12,13): 40
}

# === Step 3: Lift 2D to 3D ===
joints_3d = np.hstack([joints_2d, np.zeros((joints_2d.shape[0], 1))])
joints_3d[1, 2] = 0  # Anchor neck z = 0

for (i, j) in edges:
    x1, y1 = joints_3d[i, 0], joints_3d[i, 1]
    x2, y2 = joints_3d[j, 0], joints_3d[j, 1]
    d2d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    L = bone_lengths_cm.get((i,j)) or bone_lengths_cm.get((j,i))
    if L is None or d2d == 0:
        continue
    dz = np.sqrt(max(0, L**2 - d2d**2))  # ensure real number
    joints_3d[j, 2] = joints_3d[i, 2] + dz

# === Step 4: Plot the lifted 3D pose ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for i, j in edges:
    ax.plot([joints_3d[i, 0], joints_3d[j, 0]],
            [joints_3d[i, 1], joints_3d[j, 1]],
            [joints_3d[i, 2], joints_3d[j, 2]], 'go-')
ax.set_title("3D Pose from Annotated 2D Skeleton (output.jpg)")
ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.set_zlabel("Z (depth, cm)")
ax.view_init(elev=20, azim=-70)
plt.tight_layout()
plt.show()
