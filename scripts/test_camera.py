"""
scripts/test_camera.py
=======================
Quick sanity check: opens webcam, draws MediaPipe hand landmarks.
Run this before Phase 1 to confirm your camera + mediapipe work correctly.

Usage:
    python scripts/test_camera.py
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera. Check your webcam.")
    exit(1)

print("[INFO] Camera opened. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
        cv2.putText(frame, "HAND DETECTED ✓", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    else:
        cv2.putText(frame, "Show your hand...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

    cv2.imshow("Camera Test — AI Sign Bridge", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Camera test complete.")
