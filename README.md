# 🤟 AI Sign Bridge

> **Real-Time Indian Sign Language ↔ Speech Translator**  
> Fully offline · PyQt6 · MediaPipe · scikit-learn · pyttsx3 · Vosk

---

## Quick Start

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Test camera (do this first!)
python scripts/test_camera.py

# 3. Collect data for each sign (repeat for every sign)
python scripts/collect_data.py --label "Hello" --output data/landmarks.csv
python scripts/collect_data.py --label "Thank You" --output data/landmarks.csv
# ... (see guide for all signs)

# 4. Train the model
python scripts/train_model.py

# 5. Download Vosk model → extract to models/vosk-model/
#    https://alphacephei.com/vosk/models → vosk-model-small-en-us-0.15

# 6. Launch the app
python app/main.py
```

## Project Structure

```
ai-sign-bridge/
├── app/                   # PyQt6 application
│   ├── main.py            # Entry point
│   ├── main_window.py     # UI layout
│   ├── sign_detector.py   # MediaPipe + ML inference (QThread)
│   ├── speech_engine.py   # pyttsx3 TTS wrapper
│   └── speech_listener.py # Vosk STT (QThread)
├── data/
│   └── landmarks.csv      # Hand landmark dataset (you create this)
├── models/
│   ├── gesture_model.pkl  # Trained classifier (after training)
│   ├── label_encoder.pkl  # Label encoder (after training)
│   └── vosk-model/        # Vosk offline STT model (you download this)
├── signs/                 # GIFs for Speech→Sign pipeline
│   └── hello.gif          # One file per sign word
├── scripts/
│   ├── test_camera.py     # Sanity check
│   ├── collect_data.py    # Phase 1: dataset builder
│   └── train_model.py     # Phase 2: model trainer
└── requirements.txt
```

## Signs to Collect

| Sign | Command |
|------|---------|
| Hello | `python scripts/collect_data.py --label "Hello" --output data/landmarks.csv` |
| Thank You | `python scripts/collect_data.py --label "Thank You" --output data/landmarks.csv` |
| Yes | `python scripts/collect_data.py --label "Yes" --output data/landmarks.csv` |
| No | `python scripts/collect_data.py --label "No" --output data/landmarks.csv` |
| Help | `python scripts/collect_data.py --label "Help" --output data/landmarks.csv` |
| Water | `python scripts/collect_data.py --label "Water" --output data/landmarks.csv` |
| Please | `python scripts/collect_data.py --label "Please" --output data/landmarks.csv` |
| Sorry | `python scripts/collect_data.py --label "Sorry" --output data/landmarks.csv` |
| Good | `python scripts/collect_data.py --label "Good" --output data/landmarks.csv` |
| Stop | `python scripts/collect_data.py --label "Stop" --output data/landmarks.csv` |

**Press S to save a frame · Press Q to quit**  
Aim for **150–200 frames per sign**.

## Vosk Model Download

1. Visit: https://alphacephei.com/vosk/models
2. Download: `vosk-model-small-en-us-0.15` (~50 MB)
3. Extract the folder inside and rename/place it as: `models/vosk-model/`

## Signs GIFs

Place GIF or PNG files in the `signs/` folder named after the word:
- `signs/hello.gif`
- `signs/thank_you.gif`
- `signs/water.png`
- etc.

Sources: record yourself, or use ASL dictionaries online.
