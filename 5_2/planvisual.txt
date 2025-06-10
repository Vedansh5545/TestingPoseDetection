┌──────────────────────────────────────────────┐
│            Start: Input RGB Image            │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│   2D Keypoint Detection (MediaPipe or HRNet) │
│       → Outputs (x, y, confidence) per joint │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Semantic Bone Labeling                      │
│  → Map joint pairs to bone names             │
│  (e.g., (5,6) → left_upper_arm)              │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Visibility Mask Creation                    │
│  → Use confidence scores to mark missing     │
│     or visible bones                         │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Missing Bone Inference                      │
│  → Use symmetry & known ratios to estimate   │
│     missing joints                           │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  2D Pose Graph Completion                    │
│  → Build a complete labeled graph            │
│     with 28 joints and 34 edges              │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  3D Pose Estimation Model                    │
│  (GCN + Transformer + Decoder)               │
│  → Outputs (x, y, z) for all 28 joints       │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│        Final Output: 3D Human Skeleton       │
│   (Visualizable with labels + structure)     │
└──────────────────────────────────────────────┘
