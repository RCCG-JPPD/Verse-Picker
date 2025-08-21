# Verse Predictor Setup

This project listens to audio from your microphone, transcribes it with Whisper, and uses an LLM to predict the Bible verses being referenced. Predictions are stored in `predictions.json`.

---

## 0) Install ffmpeg

The project requires `ffmpeg` for audio processing.

- **macOS (Homebrew):**
  ```bash
  brew install ffmpeg

	•	Windows:

# Option A: winget
winget install Gyan.FFmpeg

# Option B: chocolatey
choco install ffmpeg

Or download manually and add the bin folder to your PATH.

	•	Linux (Debian/Ubuntu):

sudo apt update
sudo apt install ffmpeg



⸻

1) Python dependencies

Create a virtual environment and install required packages.
	•	macOS/Linux (bash/zsh):

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


	•	Windows (PowerShell or cmd):

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt



⸻

2) Install and configure Ollama (for LLM inference)

Ollama runs local LLMs such as LLaMA and DeepSeek.
	•	Install Ollama:
	•	macOS: Download
	•	Windows: Download
	•	Linux: Download
	•	Pull a model:

ollama pull llama3.1:8b
# or
ollama pull deepseek-r1:14b


	•	Update config.yaml:

llm:
  provider: ollama
  endpoint: http://127.0.0.1:11434
  model: deepseek-r1:14b



⸻

3) Add Bible data

Place your Bible XML files under the ./bibles folder. Example:

bibles:
  - bibles/kjv.xml
  - bibles/niv.xml


⸻

4) Run

With the virtual environment activated, start the predictor:

python verse_predictor.py --config config.yaml


⸻

Notes
	•	Make sure your microphone is enabled and Python has permission to access it:
	•	macOS: System Settings → Privacy & Security → Microphone
	•	Windows: Settings → Privacy & Security → Microphone → Allow access for python.exe
	•	Linux: Ensure PulseAudio/PipeWire is running and the input device is not muted.
	•	The output file predictions.json will contain the last 5 verse predictions ranked by likelihood.

---
