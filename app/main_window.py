"""
app/main_window.py
==================
Main PyQt6 window for AI Sign Bridge.
Layout: Webcam feed (left) | Sign output + GIF (right) | Status bar (bottom)
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QStatusBar, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtGui import QPixmap, QFont, QMovie, QColor, QImage

from app.sign_detector import SignDetector
from app.speech_engine import SpeechEngine
from app.speech_listener import SpeechListener


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0a0a0f;
    color: #e8e8f0;
    font-family: 'Segoe UI', sans-serif;
}

QFrame#card {
    background-color: #12121c;
    border: 1px solid #2a2a40;
    border-radius: 16px;
}

QLabel#section_title {
    color: #7c7cff;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
}

QLabel#webcam_label {
    background-color: #0d0d18;
    border-radius: 12px;
    border: 1px solid #1e1e30;
}

QLabel#prediction_text {
    color: #ffffff;
    font-size: 52px;
    font-weight: bold;
    letter-spacing: -1px;
}

QLabel#confidence_text {
    color: #7c7cff;
    font-size: 14px;
}

QLabel#sign_gif_label {
    background-color: #0d0d18;
    border-radius: 12px;
    border: 1px dashed #2a2a40;
    color: #3a3a5c;
    font-size: 13px;
}

QLabel#transcript_label {
    color: #a0a0c0;
    font-size: 14px;
    padding: 4px 8px;
}

QPushButton#start_btn {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #6c63ff, stop:1 #9b59b6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#start_btn:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #574fd6, stop:1 #8044a3);
}
QPushButton#start_btn:disabled {
    background: #2a2a40;
    color: #555575;
}

QPushButton#stop_btn {
    background: #1e1e30;
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#stop_btn:hover {
    background: #ff6b6b;
    color: white;
}
QPushButton#stop_btn:disabled {
    background: #12121c;
    color: #3a3a5c;
    border-color: #2a2a40;
}

QStatusBar {
    background-color: #0a0a0f;
    color: #5a5a8a;
    border-top: 1px solid #1a1a2e;
    font-size: 12px;
    padding: 4px 8px;
}

QLabel#live_badge {
    background-color: #ff4757;
    color: white;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#header_title {
    color: #ffffff;
    font-size: 22px;
    font-weight: bold;
    letter-spacing: -0.5px;
}

QLabel#history_item {
    color: #7070a0;
    font-size: 13px;
    padding: 2px 0px;
}
"""


class MainWindow(QMainWindow):
    SIGNS_DIR = "signs"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Sign Bridge — Real-Time ISL Translator")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)

        self._detector = None
        self._listener = None
        self._speech_engine = SpeechEngine()
        self._history = []
        self._current_movie = None

        self._build_ui()
        self._status("Ready. Press Start to begin.")

    # ------------------------------------------------------------------ UI BUILD

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # ── Header ──
        header = self._make_header()
        root.addLayout(header)

        # ── Main Content ──
        content = QHBoxLayout()
        content.setSpacing(16)

        left_card = self._make_left_card()    # Webcam
        right_card = self._make_right_card()  # Predictions + GIF

        content.addWidget(left_card, 3)
        content.addWidget(right_card, 2)
        root.addLayout(content, 1)

        # ── Controls ──
        controls = self._make_controls()
        root.addLayout(controls)

        # ── Status Bar ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _make_header(self):
        layout = QHBoxLayout()
        title = QLabel("🤟 AI Sign Bridge")
        title.setObjectName("header_title")
        subtitle = QLabel("  Indian Sign Language ↔ Speech Translator")
        subtitle.setStyleSheet("color: #5a5a8a; font-size: 13px;")

        self.live_badge = QLabel("⏸ IDLE")
        self.live_badge.setObjectName("live_badge")
        self.live_badge.setStyleSheet(
            "background-color: #2a2a40; color: #5a5a8a; border-radius: 8px;"
            "padding: 2px 10px; font-size: 11px; font-weight: bold;"
        )
        self.live_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        layout.addWidget(self.live_badge)
        return layout

    def _make_left_card(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        section_lbl = QLabel("LIVE CAMERA")
        section_lbl.setObjectName("section_title")

        self.webcam_label = QLabel("Camera feed will appear here")
        self.webcam_label.setObjectName("webcam_label")
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setMinimumSize(480, 360)
        self.webcam_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.fps_label = QLabel("FPS: —")
        self.fps_label.setStyleSheet("color: #3a3a5c; font-size: 11px;")

        layout.addWidget(section_lbl)
        layout.addWidget(self.webcam_label, 1)
        layout.addWidget(self.fps_label, alignment=Qt.AlignmentFlag.AlignRight)
        return card

    def _make_right_card(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)

        # ── Sign Detection Section ──
        detected_lbl = QLabel("DETECTED SIGN")
        detected_lbl.setObjectName("section_title")

        self.prediction_text = QLabel("—")
        self.prediction_text.setObjectName("prediction_text")
        self.prediction_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_text.setWordWrap(True)

        self.confidence_text = QLabel("Waiting for gesture...")
        self.confidence_text.setObjectName("confidence_text")
        self.confidence_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #1e1e30;")

        # ── Speech→Sign Section ──
        speech_lbl = QLabel("SPEECH → SIGN")
        speech_lbl.setObjectName("section_title")

        self.sign_gif_label = QLabel("Speak a word to see its sign")
        self.sign_gif_label.setObjectName("sign_gif_label")
        self.sign_gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_gif_label.setMinimumHeight(160)

        # ── Transcript ──
        transcript_lbl = QLabel("MIC TRANSCRIPT")
        transcript_lbl.setObjectName("section_title")

        self.transcript_label = QLabel("—")
        self.transcript_label.setObjectName("transcript_label")
        self.transcript_label.setWordWrap(True)

        # ── History ──
        history_lbl = QLabel("HISTORY")
        history_lbl.setObjectName("section_title")
        self.history_label = QLabel("")
        self.history_label.setObjectName("history_item")
        self.history_label.setWordWrap(True)

        layout.addWidget(detected_lbl)
        layout.addWidget(self.prediction_text)
        layout.addWidget(self.confidence_text)
        layout.addWidget(divider)
        layout.addWidget(speech_lbl)
        layout.addWidget(self.sign_gif_label, 1)
        layout.addWidget(transcript_lbl)
        layout.addWidget(self.transcript_label)
        layout.addWidget(history_lbl)
        layout.addWidget(self.history_label)
        layout.addStretch()
        return card

    def _make_controls(self):
        layout = QHBoxLayout()
        layout.setSpacing(12)

        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.setFixedHeight(44)
        self.start_btn.clicked.connect(self._start_detection)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setFixedHeight(44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_detection)

        hint = QLabel("Sign → Speech  ·  Speech → Sign  ·  Fully Offline")
        hint.setStyleSheet("color: #3a3a5c; font-size: 12px;")

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addStretch()
        layout.addWidget(hint)
        return layout

    # ------------------------------------------------------------------ ACTIONS

    def _start_detection(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_live(True)
        self._status("Detection running…")

        # Sign→Speech thread
        self._detector = SignDetector()
        self._detector.frame_ready.connect(self._update_frame)
        self._detector.prediction_ready.connect(self._on_prediction)
        self._detector.start()

        # Speech→Sign thread
        self._listener = SpeechListener()
        self._listener.word_recognized.connect(self._on_word_recognized)
        self._listener.start()

    def _stop_detection(self):
        if self._detector:
            self._detector.stop()
            self._detector = None
        if self._listener:
            self._listener.stop()
            self._listener = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_live(False)
        self._status("Stopped.")
        self.webcam_label.setText("Camera feed will appear here")
        self.webcam_label.setPixmap(QPixmap())

    # ------------------------------------------------------------------ SLOTS

    @pyqtSlot(QImage)
    def _update_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.webcam_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.webcam_label.setPixmap(scaled)

    @pyqtSlot(str, float)
    def _on_prediction(self, label: str, confidence: float):
        self.prediction_text.setText(label)
        pct = int(confidence * 100)
        self.confidence_text.setText(f"Confidence: {pct}%")

        # Update history
        self._history.append(label)
        if len(self._history) > 5:
            self._history.pop(0)
        self.history_label.setText("  ·  ".join(self._history))

        # Speak the word
        self._speech_engine.speak(label)
        self._status(f"Detected: {label} ({pct}%)")

    @pyqtSlot(str)
    def _on_word_recognized(self, word: str):
        self.transcript_label.setText(f'"{word}"')
        self._status(f"Heard: '{word}' — showing sign…")
        self._show_sign_gif(word)

    # ------------------------------------------------------------------ HELPERS

    def _show_sign_gif(self, word: str):
        """Try to display a GIF/image for the recognized word."""
        # Search for matching file: signs/hello.gif or signs/hello.png
        candidates = [
            os.path.join(self.SIGNS_DIR, f"{word.lower()}.gif"),
            os.path.join(self.SIGNS_DIR, f"{word.lower().replace(' ', '_')}.gif"),
            os.path.join(self.SIGNS_DIR, f"{word.lower()}.png"),
            os.path.join(self.SIGNS_DIR, f"{word.lower().replace(' ', '_')}.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                if path.endswith(".gif"):
                    if self._current_movie:
                        self._current_movie.stop()
                    movie = QMovie(path)
                    self._current_movie = movie
                    self.sign_gif_label.setMovie(movie)
                    movie.start()
                else:
                    self.sign_gif_label.setPixmap(
                        QPixmap(path).scaled(200, 160, Qt.AspectRatioMode.KeepAspectRatio)
                    )
                return

        # No file found
        self.sign_gif_label.setText(f'No sign found for "{word}"\nAdd {word.lower()}.gif to /signs/')

    def _set_live(self, is_live: bool):
        if is_live:
            self.live_badge.setText("● LIVE")
            self.live_badge.setStyleSheet(
                "background-color: #ff4757; color: white; border-radius: 8px;"
                "padding: 2px 10px; font-size: 11px; font-weight: bold;"
            )
        else:
            self.live_badge.setText("⏸ IDLE")
            self.live_badge.setStyleSheet(
                "background-color: #2a2a40; color: #5a5a8a; border-radius: 8px;"
                "padding: 2px 10px; font-size: 11px; font-weight: bold;"
            )

    def _status(self, msg: str):
        self.status_bar.showMessage(f"  {msg}")

    def closeEvent(self, event):
        self._stop_detection()
        super().closeEvent(event)
