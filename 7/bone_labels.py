# bone_labels.py

from skeleton_utils import MPIINF_EDGES

# 28 joint names in your MPI-INF order:
JOINT_NAMES = [
    "pelvis",        # 0
    "right_hip",     # 1
    "right_knee",    # 2
    "right_ankle",   # 3
    "right_heel",    # 4
    "left_hip",      # 5
    "left_knee",     # 6
    "left_ankle",    # 7
    "left_heel",     # 8
    "spine_mid",     # 9
    "neck",          #10
    "head_top",      #11
    "left_shoulder", #12
    "left_elbow",    #13
    "left_wrist",    #14
    "right_shoulder",#15
    "right_elbow",   #16
    "right_wrist",   #17
    # Fill the rest if you use >18 joints; otherwise these will be unused:
    "joint_18","joint_19","joint_20","joint_21",
    "joint_22","joint_23","joint_24","joint_25",
    "joint_26","joint_27"
]

# Build a dict mapping each bone (i,j) → "name1→name2"
BONE_LABELS = {
    (i,j): f"{JOINT_NAMES[i]}→{JOINT_NAMES[j]}"
    for (i,j) in MPIINF_EDGES
}

# Allow reverse lookup too
BONE_LABELS_SYM = {
    **BONE_LABELS,
    **{(j,i): name for (i,j), name in BONE_LABELS.items()}
}
