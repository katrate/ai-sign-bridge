"""
Download ASL landmark dataset from HuggingFace.
"""
import os, sys
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

print("[INFO] Trying to download ASL dataset from HuggingFace...")

try:
    from datasets import load_dataset
    print("[INFO] Downloading xeno-nova/asl-sign-language-recognizer...")
    ds = load_dataset("xeno-nova/asl-sign-language-recognizer", split="train")
    df = ds.to_pandas()
    print(f"[INFO] Downloaded {df.shape[0]} rows. Columns: {list(df.columns)}")

    label_col = None
    for c in ["label", "Label", "class", "sign", "gesture"]:
        if c in df.columns:
            label_col = c
            break
    if not label_col:
        label_col = df.columns[-1]
    labels = df[label_col].values

    x_cols = sorted([c for c in df.columns if c.startswith("x") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    y_cols = sorted([c for c in df.columns if c.startswith("y") and c[1:].isdigit()], key=lambda c: int(c[1:]))

    if len(x_cols) == 21 and len(y_cols) == 21:
        coords = []
        for xi, yi in zip(x_cols, y_cols):
            coords.append(df[xi].values)
            coords.append(df[yi].values)
        X = np.column_stack(coords).astype(np.float32)
    else:
        numeric_cols = [c for c in df.columns if c != label_col]
        X = df[numeric_cols[:42]].values.astype(np.float32)

    out = pd.DataFrame(X)
    out["label"] = labels
    out.to_csv("data/landmarks.csv", index=False, header=False)
    print(f"[SUCCESS] Saved {len(out)} rows to data/landmarks.csv")
    print(f"[INFO] Labels: {sorted(set(labels))}")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Failed: {e}")
    sys.exit(1)
