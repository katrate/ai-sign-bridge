"""
Phase 1: Data Collection Script
================================
Captures hand landmarks from webcam and saves them to a CSV file.
Auto-saves every frame while SPACE is held. 

Usage:
    python scripts/collect_data.py

Controls:
    SPACE - Hold to auto-save current frames (no need to press repeatedly)
    Q     - Quit
"""

import cv2
import mediapipe as mp
import csv
import os
import time

# ------------ MediaPipe Setup ------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ------------ Webcam ------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

OUTPUT_FILE = "data/my_signs.csv"
os.makedirs("data", exist_ok=True)

# Labels: all 26 letters + 10 digits
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")

label_idx = 0
current_label = LABELS[label_idx]
count_per_label = {}
total_saved = 0
recording = False
last_save_time = 0
SAVE_INTERVAL = 0.05  # 20 frames per second when holding space

# Open CSV in append mode
csv_file = open(OUTPUT_FILE, "a", newline="")
writer = csv.writer(csv_file)

print(f"\n[AI Sign Bridge] Data Collector")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Hold SPACE to record frames for current sign")
print(f"  N = Next sign, P = Previous sign, Q = Quit\n")
print(f"  CURRENT SIGN: [{current_label}]")

def extract_features(hand_landmarks):
    """Extract 63 raw XYZ features from hand landmarks."""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_row = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
            landmark_row = extract_features(hand_landmarks)

    # Auto-save while SPACE is held
    if recording and landmark_row:
        now = time.time()
        if now - last_save_time >= SAVE_INTERVAL:
            writer.writerow(landmark_row + [current_label])
            total_saved += 1
            count_per_label[current_label] = count_per_label.get(current_label, 0) + 1
            last_save_time = now

    # UI Overlay
    count_this = count_per_label.get(current_label, 0)
    color = (0, 200, 100) if recording else (0, 120, 255)
    status = "RECORDING..." if recording else "Hold SPACE to record"

    cv2.rectangle(frame, (0, 0), (640, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"Sign: [{current_label}]  Frames: {count_this}  Total: {total_saved}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if landmark_row:
        cv2.putText(frame, "HAND DETECTED", (480, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)

    cv2.putText(frame, "N=Next  P=Prev  Q=Quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("AI Sign Bridge — Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        label_idx = min(label_idx + 1, len(LABELS) - 1)
        current_label = LABELS[label_idx]
        recording = False
        print(f"  -> Next sign: [{current_label}]  (saved so far: {count_per_label.get(current_label, 0)})")
    elif key == ord('p'):
        label_idx = max(label_idx - 1, 0)
        current_label = LABELS[label_idx]
        recording = False
        print(f"  -> Prev sign: [{current_label}]  (saved so far: {count_per_label.get(current_label, 0)})")
    elif key == 32:  # SPACE pressed
        recording = True
    else:
        if recording:
            print(f"  [DONE] Saved {count_per_label.get(current_label,0)} frames for '{current_label}'")
        recording = False

csv_file.close()
cap.release()
cv2.destroyAllWindows()

print(f"\n[DONE] Saved {total_saved} total frames to '{OUTPUT_FILE}'")
print(f"  Per-sign summary: {count_per_label}")
print(f"\nNEXT: Run: python scripts/prepare_dataset.py --input data/my_signs.csv")
