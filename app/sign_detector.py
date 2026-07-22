"""
app/sign_detector.py
=====================
Runs in a QThread. Captures webcam frames, extracts MediaPipe hand landmarks,
predicts the sign using the trained model, and emits signals for the UI.
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import math
from collections import Counter
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

class SignDetector(QThread):
    # Signals emitted to the main UI thread
    frame_ready = pyqtSignal(QImage)          # Processed webcam frame
    prediction_ready = pyqtSignal(str, float) # (sign_label, confidence)

    MODEL_PATH = "models/gesture_model.pkl"
    ENCODER_PATH = "models/label_encoder.pkl"
    BUFFER_SIZE = 5  # Frames needed before confirming a prediction

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.model = None
        self.label_encoder = None
        self._load_model()
        self._buffer = []
        self._last_spoken = None

    def _load_model(self):
        try:
            self.model = joblib.load(self.MODEL_PATH)
            self.label_encoder = joblib.load(self.ENCODER_PATH)
            print("[SignDetector] Model loaded successfully.")
        except FileNotFoundError:
            print("[SignDetector] WARNING: Model not found. Train the model first.")

    def run(self):
        self.running = True

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks and self.model:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw styled landmarks
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style()
                    )

                    # Extract 42 features (X, Y only, relative to wrist for translation invariance)
                    # Extract 63 features: X, Y, Z for all 21 landmarks
                    # with Bounding Box Normalization (position + scale invariant)
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    z_coords = [lm.z for lm in hand_landmarks.landmark]

                    def bbox_norm(vals):
                        mn, mx = min(vals), max(vals)
                        r = mx - mn if mx > mn else 1.0
                        return [(v - mn) / r for v in vals]

                    x_n = bbox_norm(x_coords)
                    y_n = bbox_norm(y_coords)
                    z_n = bbox_norm(z_coords)

                    row = []
                    for x, y, z in zip(x_n, y_n, z_n):
                        row.extend([x, y, z])

                    # Predict
                    features = np.array(row).reshape(1, -1)
                    proba = self.model.predict_proba(features)[0]
                    pred_idx = np.argmax(proba)
                    confidence = float(proba[pred_idx])
                    label = self.label_encoder.inverse_transform([pred_idx])[0]

                    # Smoothing buffer
                    self._buffer.append(label)
                    if len(self._buffer) > self.BUFFER_SIZE:
                        self._buffer.pop(0)

                    if len(self._buffer) == self.BUFFER_SIZE:
                        most_common, freq = Counter(self._buffer).most_common(1)[0]
                        if freq >= self.BUFFER_SIZE * 0.6:  # 60% agreement
                            if most_common != self._last_spoken and confidence > 0.60:
                                self._last_spoken = most_common
                                self.prediction_ready.emit(most_common, confidence)
                                self._buffer.clear()  # Start fresh for the next sign
            else:
                self._buffer.clear()
                self._last_spoken = None

            # Convert frame to QImage for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qt_image)

        cap.release()
        hands.close()

    def stop(self):
        self.running = False
        self.wait()
