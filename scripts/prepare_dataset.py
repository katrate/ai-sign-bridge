"""
scripts/prepare_dataset.py
===========================
Prepares a landmarks CSV (from collect_data.py or Kaggle) into a 
normalized feature set for training.

Usage:
    python scripts/prepare_dataset.py --input data/my_signs.csv
    python scripts/prepare_dataset.py --input data/recorded_keypoints.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to raw landmarks CSV")
parser.add_argument("--output", type=str, default="data/landmarks.csv", help="Output path")
args = parser.parse_args()

if not os.path.exists(args.input):
    print(f"[ERROR] File not found: {args.input}")
    sys.exit(1)

print(f"[INFO] Loading: {args.input}")
df = pd.read_csv(args.input, header=None)
print(f"\n{'='*50}")
print(f"  Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"{'='*50}")

n_cols = df.shape[1]

# ── Detect format ──────────────────────────────────────────────
# Format A: Our own collect_data.py   → 63 coords + 1 label  = 64 cols
# Format B: Kaggle ISL dataset        → 91 cols (label + metadata + coords)

if n_cols == 64:
    print("[INFO] Detected: collect_data.py format (63 features + 1 label)")
    labels = df.iloc[:, -1].values
    features_raw = df.iloc[:, :63].values  # x,y,z for 21 landmarks
    X_3d = features_raw.reshape(-1, 21, 3)

elif n_cols == 91:
    print("[INFO] Detected: Kaggle ISL dataset format (91 columns)")
    labels = df.iloc[:, 0].values
    handedness = df.iloc[:, 4].values
    features_list = []
    for i in range(len(df)):
        if handedness[i] == 'L':
            xy = df.iloc[i, 5:47].values
        else:
            xy = df.iloc[i, 49:91].values
        # Pad Z=0 since Kaggle has no depth
        xyz = np.concatenate([xy.reshape(21, 2), np.zeros((21, 1))], axis=1)
        features_list.append(xyz)
    X_3d = np.array(features_list, dtype=np.float32)  # (N, 21, 3)

else:
    # Generic fallback: last column is label, rest are features
    print(f"[INFO] Generic format detected. Using last column as label.")
    labels = df.iloc[:, -1].values
    features_raw = df.iloc[:, :-1].values.astype(np.float32)
    n_features = features_raw.shape[1]
    if n_features >= 63:
        X_3d = features_raw[:, :63].reshape(-1, 21, 3)
    elif n_features >= 42:
        xy = features_raw[:, :42].reshape(-1, 21, 2)
        X_3d = np.concatenate([xy, np.zeros((*xy.shape[:2], 1))], axis=2)
    else:
        print(f"[ERROR] Not enough features ({n_features}). Need at least 42.")
        sys.exit(1)

print(f"[INFO] Labels: {sorted(set(labels))}")
print(f"[INFO] Raw 3D shape: {X_3d.shape}")


def normalize_hand(X_3d_batch):
    """
    Apply bounding box normalization per-sample per-axis.
    Scales every hand into the [0,1] cube — invariant to position, scale, distance.
    """
    min_vals = X_3d_batch.min(axis=1, keepdims=True)  # (N, 1, 3)
    max_vals = X_3d_batch.max(axis=1, keepdims=True)  # (N, 1, 3)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    return (X_3d_batch - min_vals) / ranges


# Normalize
X_norm = normalize_hand(X_3d)

# Mirror augmentation (left/right hand invariance) — flip X axis
X_flip = X_norm.copy()
X_flip[:, :, 0] = 1.0 - X_flip[:, :, 0]  # mirror X

X_combined = np.concatenate([X_norm, X_flip], axis=0)
y_combined = np.concatenate([labels, labels], axis=0)

# Flatten to (N, 63)
X_flat = X_combined.reshape(-1, 63)

# Drop NaN rows
mask = ~np.isnan(X_flat).any(axis=1)
X_flat = X_flat[mask]
y_combined = y_combined[mask]

print(f"[INFO] Final dataset: {X_flat.shape[0]} rows x {X_flat.shape[1]} features (after augmentation)")

# Save
os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
out_df = pd.DataFrame(X_flat)
out_df['label'] = y_combined
out_df.to_csv(args.output, index=False, header=False)

print(f"\n[SUCCESS] Saved {len(out_df)} rows to '{args.output}'")
print(f"[NEXT] Run: python scripts/train_model.py")
