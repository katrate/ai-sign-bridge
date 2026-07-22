"""
Phase 2: Model Training Script
================================
Loads landmark CSV data, trains a RandomForest classifier,
evaluates accuracy, and saves the model.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --input data/landmarks_augmented.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="data/landmarks.csv",
                    help="Path to landmarks CSV (default: data/landmarks.csv)")
args = parser.parse_args()

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ------------ Config ------------
DATA_PATH = args.input
MODEL_PATH = "models/gesture_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# ------------ Load Data ------------
print("[INFO] Loading dataset...")
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'.")
    exit(1)

df = pd.read_csv(DATA_PATH, header=None)
print(f"[INFO] Loaded {len(df)} rows.")
print(f"[INFO] Signs found: {df.iloc[:, -1].unique()}")

X = df.iloc[:, :-1].values   # 63 landmark features
y = df.iloc[:, -1].values    # Sign label strings

# ------------ Encode Labels ------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"[INFO] Classes: {le.classes_}")

# ------------ Train/Test Split ------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

# ------------ Train Model ------------
print("[INFO] Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ------------ Evaluate ------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Test Accuracy: {acc*100:.2f}%")

print("\n[RESULT] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ------------ Save Model & Encoder ------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\n[SAVED] Model   -> {MODEL_PATH}")
print(f"[SAVED] Encoder -> {ENCODER_PATH}")

if acc < 0.90:
    print("\n[WARNING] Accuracy below 90%. Consider:")
    print("  1. Collecting more data per sign (aim for 200+ frames each)")
    print("  2. Varying hand angles while collecting")
    print("  3. Trying SVC: change model to SVC(kernel='rbf', C=10)")
else:
    print("\n[SUCCESS] Model is ready for real-time inference!")
