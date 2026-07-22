"""
app/speech_engine.py
=====================
Thin wrapper around pyttsx3 for offline text-to-speech.
Runs TTS in a separate thread to avoid blocking the UI.
"""

import pyttsx3
import threading


class SpeechEngine:
    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', 145)   # Speaking speed
        self._engine.setProperty('volume', 1.0) # Max volume
        self._lock = threading.Lock()

        # Try to set a clearer voice
        voices = self._engine.getProperty('voices')
        for v in voices:
            if 'zira' in v.name.lower() or 'david' in v.name.lower():
                self._engine.setProperty('voice', v.id)
                break

    def speak(self, text: str):
        """Speak the given text in a non-blocking background thread."""
        def _speak():
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()
