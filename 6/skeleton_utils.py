# skeleton_utils.py

# 1) List of (i,j) edges for your 28-joint MPI-INF-3DHP skeleton:
MPIINF_EDGES = [
    (0,1), (1,2), (2,3), (3,4),      # right leg
    (0,5), (5,6), (6,7), (7,8),      # left leg
    (0,9), (9,10), (10,11),          # spine → neck → head
    (11,12), (12,13), (13,14),       # left arm
    (11,15), (15,16), (16,17),       # right arm
    (9,18), (9,19)                   # shoulders → hips (optional torso edges)
    # …add or prune to match your 34 total bones
]

# 2) Mapping from MediaPipe’s 33-landmark indices → MPI-INF-3DHP [0..27].
#    Fill in the actual values for your dataset!
MEDIAPIPE_TO_MPIINF = {
    0:  9,   # Nose → Neck
    11: 12,  # Left Shoulder → MPI index for left shoulder
    12: 15,  # Right Shoulder → MPI index for right shoulder
    # …continue for all 33 MediaPipe points; unmapped entries → None or skip
}
