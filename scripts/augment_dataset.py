"""
scripts/augment_dataset.py
===========================
Takes the existing landmarks.csv and generates a highly augmented version
by simulating many different people's hand shapes, angles, and sizes.

This improves generalization so the model works for ANY user without retraining.

Augmentations applied:
  - 2D rotation (hand rotated -30 to +30 degrees around palm center)
  - Gaussian noise (simulates measurement jitter)
  - Scaling variation (simulates different hand sizes)
  - Aspect ratio stretch (simulates different camera heights/distances)

Usage:
    python scripts/augment_dataset.py
"""

import numpy as np
import pandas as pd
import os

INPUT  = "data/landmarks.csv"
OUTPUT = "data/landmarks_augmented.csv"
N_AUGMENTS = 8   # Generate 8 extra variants per original sample

print(f"[INFO] Loading {INPUT}...")
df = pd.read_csv(INPUT, header=None)
X_raw = df.iloc[:, :-1].values.astype(np.float32)
y_raw = df.iloc[:, -1].values

print(f"[INFO] Original dataset: {X_raw.shape[0]} rows x {X_raw.shape[1]} features")

# Reshape to (N, 21, 2)
X_2d = X_raw.reshape(-1, 21, 2)

def rotate_2d(points_xy, angle_rad):
    """Rotate 2D points around their centroid."""
    cx, cy = points_xy.mean(axis=0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    centered = points_xy - np.array([cx, cy])
    rotated = centered @ R.T
    return rotated + np.array([cx, cy])

def bbox_normalize(X_2d_batch):
    """Bounding box normalize per sample while preserving aspect ratio."""
    mn = X_2d_batch.min(axis=1, keepdims=True)  # (N, 1, 2)
    mx = X_2d_batch.max(axis=1, keepdims=True)  # (N, 1, 2)
    rng = mx - mn  # (N, 1, 2)
    max_range = np.max(rng, axis=2, keepdims=True)  # (N, 1, 1)
    max_range[max_range == 0] = 1.0
    return (X_2d_batch - mn) / max_range

def augment_batch(X_2d, n_aug=8):
    N = X_2d.shape[0]
    all_augmented = []

    for _ in range(n_aug):
        aug = X_2d.copy()

        # 1. Random 2D rotation (-30 to +30 degrees)
        angles = np.random.uniform(-np.pi/6, np.pi/6, N)
        for i in range(N):
            xy = aug[i, :, :]
            aug[i, :, :] = rotate_2d(xy, angles[i])

        # 2. Random scale (simulate different hand sizes / distances)
        scales = np.random.uniform(0.85, 1.15, (N, 1, 1))
        centers = aug.mean(axis=1, keepdims=True)
        aug = centers + (aug - centers) * scales

        # 3. Random aspect ratio stretch (simulate different camera angles)
        stretch_x = np.random.uniform(0.88, 1.12, (N, 1, 1))
        stretch_y = np.random.uniform(0.88, 1.12, (N, 1, 1))
        aug[:, :, 0] *= stretch_x[:, :, 0]
        aug[:, :, 1] *= stretch_y[:, :, 0]

        # 4. Gaussian noise (simulates landmark detection jitter)
        aug += np.random.normal(0, 0.008, aug.shape).astype(np.float32)

        # 5. Re-normalize after augmentation
        aug = bbox_normalize(aug)

        all_augmented.append(aug)

    return np.concatenate(all_augmented, axis=0)

print(f"[INFO] Generating {N_AUGMENTS}x augmented samples...")
X_aug = augment_batch(X_2d, N_AUGMENTS)
y_aug = np.tile(y_raw, N_AUGMENTS)

# Combine original + augmented
X_final = np.concatenate([X_2d, X_aug], axis=0)
y_final = np.concatenate([y_raw, y_aug], axis=0)

# Flatten to (N, 42)
X_flat = X_final.reshape(-1, 42)

# Drop NaN rows
mask = ~np.isnan(X_flat).any(axis=1)
X_flat = X_flat[mask]
y_final = y_final[mask]

print(f"[INFO] Final augmented dataset: {X_flat.shape[0]} rows x {X_flat.shape[1]} features")
print(f"         ({X_flat.shape[0] // len(y_raw):.1f}x original size)")

# Save
out_df = pd.DataFrame(X_flat)
out_df['label'] = y_final
out_df.to_csv(OUTPUT, index=False, header=False)

print(f"\n[SUCCESS] Saved to '{OUTPUT}'")
print(f"[NEXT] Run: python scripts/train_model.py --input data/landmarks_augmented.csv")
