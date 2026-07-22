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

# Auto-detect header: if first row has strings in numeric columns, it's a header
raw_peek = pd.read_csv(args.input, header=None, nrows=1)
first_cell = str(raw_peek.iloc[0, 0])
has_header = not first_cell.replace('.', '').replace('-', '').replace('e', '').isdigit()

if has_header:
    df = pd.read_csv(args.input)
else:
    df = pd.read_csv(args.input, header=None)

print(f"\n{'='*50}")
print(f"  Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"{'='*50}")

n_cols = df.shape[1]

# ── Detect format ──────────────────────────────────────────────
# Format A: Our own collect_data.py   → 63 coords + 1 label  = 64 cols (no header)
# Format B: Kaggle ISL dataset        → 91 cols (label + metadata + coords, no header)
# Format C: New ASL dataset (Kaggle)  → named cols x0,y0,z0...x20,y20,z20,label (with header)

# Format C: named columns with x0,y0,z0 pattern
if has_header and 'label' in df.columns and 'x0' in df.columns:
    print("[INFO] Detected: Named ASL dataset format (x0,y0,z0...x20,y20,z20 + label)")
    labels = df['label'].values
    # Extract x,y for each landmark (skip z)
    x_cols = sorted([c for c in df.columns if c.startswith('x') and c[1:].isdigit()], key=lambda c: int(c[1:]))
    y_cols = sorted([c for c in df.columns if c.startswith('y') and c[1:].isdigit()], key=lambda c: int(c[1:]))
    pts = []
    for xi, yi in zip(x_cols, y_cols):
        pts.append(df[xi].values)
        pts.append(df[yi].values)
    X_2d = np.column_stack(pts).astype(np.float32).reshape(-1, 21, 2)

elif n_cols == 64:
    print("[INFO] Detected: collect_data.py format (63 features + 1 label)")
    labels = df.iloc[:, -1].values
    features_raw = df.iloc[:, :63].values
    # Extract only X, Y raw coordinates
    X_3d = features_raw.reshape(-1, 21, 3)
    X_2d = X_3d[:, :, :2]

elif n_cols == 91:
    print("[INFO] Detected: Kaggle ISL dataset format (91 columns)")
    labels = df.iloc[:, 0].values
    handedness = df.iloc[:, 4].values
    features_list = []
    for i in range(len(df)):
        # The right hand (columns 49:91) is the signing hand in this dataset.
        # Column 48 is 'R' label, 49:91 are the 42 coordinates.
        xy = df.iloc[i, 49:91].values
        features_list.append(xy.reshape(21, 2))
    X_2d = np.array(features_list, dtype=np.float32)  # (N, 21, 2)

else:
    print(f"[INFO] Generic format detected. Using last column as label.")
    labels = df.iloc[:, -1].values
    features_raw = df.iloc[:, :-1].values.astype(np.float32)
    n_features = features_raw.shape[1]
    if n_features >= 42:
        X_2d = features_raw[:, :42].reshape(-1, 21, 2)
    else:
        print(f"[ERROR] Not enough features ({n_features}). Need at least 42.")
        sys.exit(1)

print(f"[INFO] Labels: {sorted(set(labels))}")
print(f"[INFO] Raw 2D shape: {X_2d.shape}")

# Mirror augmentation (left/right hand invariance) BEFORE normalization
X_2d_flip = X_2d.copy()
# Flip the X axis raw values by multiplying by -1. 
# Normalization will automatically shift them back to 0.
X_2d_flip[:, :, 0] = -X_2d_flip[:, :, 0]

# Combine raw original and raw flipped
X_combined_raw = np.concatenate([X_2d, X_2d_flip], axis=0)
y_combined = np.concatenate([labels, labels], axis=0)

def normalize_hand(X_2d_batch):
    """
    Apply bounding box normalization per-sample per-axis while preserving aspect ratio.
    """
    min_vals = X_2d_batch.min(axis=1, keepdims=True)  # (N, 1, 2)
    max_vals = X_2d_batch.max(axis=1, keepdims=True)  # (N, 1, 2)
    ranges = max_vals - min_vals  # (N, 1, 2)
    # Find the maximum range between width and height for each hand sample
    max_range = np.max(ranges, axis=2, keepdims=True)  # (N, 1, 1)
    max_range[max_range == 0] = 1.0
    return (X_2d_batch - min_vals) / max_range

# Normalize all data
X_norm = normalize_hand(X_combined_raw)

# Flatten to (N, 42)
X_flat = X_norm.reshape(-1, 42)

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
