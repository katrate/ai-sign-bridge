"""
app/speech_listener.py
=======================
Runs Vosk offline speech recognition in a daemon QThread.
Emits a signal with the recognized word when speech is detected.
Uses sounddevice (already installed) instead of pyaudio.
"""

import json
import os
import numpy as np
import queue

import vosk
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal


class SpeechListener(QThread):
    word_recognized = pyqtSignal(str)  # Emits recognized word(s)

    MODEL_PATH = "models/vosk-model"
    SAMPLE_RATE = 16000
    CHUNK = 4096

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self._model = None
        self._queue = queue.Queue()
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.MODEL_PATH):
            self._model = vosk.Model(self.MODEL_PATH)
            print("[SpeechListener] Vosk model loaded.")
        else:
            print(f"[SpeechListener] WARNING: Model not found at '{self.MODEL_PATH}'.")
            print("  Download from: https://alphacephei.com/vosk/models")
            print("  Extract as: models/vosk-model/")

    def _audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"[SpeechListener] Audio status: {status}")
        self._queue.put(bytes(indata))

    def run(self):
        if not self._model:
            print("[SpeechListener] No model. Speech→Sign pipeline disabled.")
            return

        self.running = True
        rec = vosk.KaldiRecognizer(self._model, self.SAMPLE_RATE)

        print("[SpeechListener] Listening...")
        with sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.CHUNK,
            dtype="int16",
            channels=1,
            callback=self._audio_callback
        ):
            while self.running:
                try:
                    data = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip().lower()
                    if text:
                        print(f"[SpeechListener] Recognized: '{text}'")
                        for word in text.split():
                            self.word_recognized.emit(word)

    def stop(self):
        self.running = False
        self.wait()
