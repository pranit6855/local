# Local AI Assistant 🎙️⚡

> A lightweight, privacy-first Windows desktop assistant inspired by Parakeet AI. Captures microphone speech, transcribes audio locally via **faster-whisper**, and generates AI answers with progressive token streaming using a locally running **Ollama LLM**. 

**100% Offline • Zero Cloud APIs • Zero Data Transmission • Zero API Keys**

---

## Overview & Architecture

The application provides a distraction-free, native desktop experience designed for high responsiveness, privacy, and low resource overhead:

```
+-------------------------------------------------------------+
|                     Local AI Assistant                      |
|                                                             |
| [● Start Recording] [■ Stop Recording]  (00:14) [||||||   ] |
+-------------------------------------------------------------+
| Transcribed Question / Prompt               [Copy]          |
| "What are the core differences between TCP and UDP?"        |
|                                         [✨ Suggest Answer] |
+-------------------------------------------------------------+
| AI Suggested Answer                         [Copy] [Stop]   |
| TCP (Transmission Control Protocol) is connection-oriented, |
| reliable, and ensures in-order delivery...                  |
+-------------------------------------------------------------+
| Ollama Online (qwen2.5:3b / llama3.2) | 100% Local Inference|
+-------------------------------------------------------------+
```

### Core Architecture Components:
1. **Desktop GUI (`PySide6`)**: Minimalist, sleek dark interface with real-time visualizers, auto-scrolling streaming answer panel, responsive splitter, and keyboard shortcuts (`Ctrl+Enter` to suggest answer).
2. **Audio Recorder (`sounddevice` + `soundfile`)**: Non-blocking microphone capture with RMS level monitoring, input device detection, and clean temporary WAV serialization.
3. **Local Speech-to-Text (`faster-whisper`)**: Powered by CTranslate2 on CPU with `int8` quantization. Fast, accurate, and completely offline.
4. **Local LLM Inference (`Ollama`)**: Local HTTP streaming client connecting to Ollama daemon (`127.0.0.1:11434`) yielding progressive token streams into the GUI without freezing the main thread.

---

## 💻 Hardware Requirements & Model Storage

| Component | Default Model | Disk Storage | RAM / VRAM | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Speech-to-Text** | Whisper `base` (CTranslate2) | ~140 MB | ~300 MB RAM | ~0.5s - 1.5s per utterance on standard CPU |
| **LLM Inference** | `qwen2.5:3b` or `llama3.2:3b` | ~1.9 GB - 2.0 GB | ~3.5 GB - 4.5 GB RAM | Real-time streaming (~25-45 tokens/sec) |

*Works on standard modern Windows laptops/desktops with 8 GB+ RAM. GPU/CUDA is optional—CPU mode is enabled and fully optimized out-of-the-box.*

---

## 🛠️ Prerequisites for Windows

1. **Windows 10 / 11** (64-bit).
2. **Python 3.11 or 3.12** installed (ensure "Add Python to PATH" was checked during setup).
3. **Working Microphone** (integrated laptop array or external USB headset).
4. **Ollama for Windows**: Download from [ollama.com/download](https://ollama.com/download).

---

## 🚀 Step-by-Step Setup Guide

### 1. Clone or Open Workspace
Open PowerShell or Command Prompt in the project folder:
```powershell
cd c:\Users\gupta\Downloads\namehtml
```

### 2. Create and Activate Virtual Environment
```powershell
# Create an isolated virtual environment
python -m venv .venv

# Activate the virtual environment
# In PowerShell:
.\.venv\Scripts\Activate.ps1

# (Or in Command Prompt: .\.venv\Scripts\activate.bat)
```

> **Note on PowerShell Execution Policy**: If PowerShell prevents script execution, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Install & Launch Local LLM (Ollama)

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Launch Ollama (it runs as a background service listening on `http://127.0.0.1:11434`).
3. Download the default lightweight instruction model (`qwen2.5:3b` or `llama3.2`):
```powershell
# Option A: Recommended Qwen 2.5 3B instruction model (~1.9 GB)
ollama pull qwen2.5:3b

# Option B: Or Llama 3.2 3B (~2.0 GB)
ollama pull llama3.2
```

To verify Ollama models on your machine:
```powershell
ollama list
```

### 5. Initialize the Whisper STT Model
The Whisper `base` model will automatically download on the first run to the local HuggingFace cache (`~/.cache/huggingface/hub/`).

To pre-initialize or verify the Whisper engine and audio setup before launching the UI:
```powershell
python test_whisper.py
```

---

## 🏃 Running the Application

Launch the desktop assistant:
```powershell
python main.py
```

### Using the Assistant:
1. **Start Recording**: Click **Start Recording** (or speak into your microphone). The duration counter and audio level visualizer will respond to your voice.
2. **Stop Recording**: Click **Stop Recording**. The application will transcribe your speech locally within 1-2 seconds.
3. **Review & Edit**: The transcribed question appears in the top panel. You can tweak or type additional text directly.
4. **Suggest Answer**: Click **✨ Suggest Answer** (or press `Ctrl+Enter`). The answer will progressively stream token-by-token from your local LLM.
5. **Copy / Reset**: Use **Copy** or **Copy Answer** to copy text, or **Reset / Clear All** for a fresh session.

---

## 🧪 Smoke Tests & Diagnostics

To execute the diagnostic test suite:
```powershell
python test_app.py
```

This verifies:
- All required modules (`PySide6`, `sounddevice`, `soundfile`, `faster_whisper`, `requests`).
- Audio input devices and microphone permissions.
- Local Ollama connectivity and model inventory.
- Token streaming from local Ollama.
- PySide6 GUI window instantiation.

---

## 📂 Project Structure

```
local-ai-assistant/
├── main.py                  # Main GUI entry point & high-DPI startup
├── requirements.txt         # Pinned production dependencies
├── README.md                # Comprehensive documentation & setup instructions
├── .gitignore               # Excludes models, caches, recordings, and venv
├── test_app.py              # Automated smoke tests for all subsystems
├── test_whisper.py          # Faster-Whisper validation script
└── app/
    ├── __init__.py          # App package declaration
    ├── ui.py                # PySide6 desktop interface (Parakeet-inspired dark UI)
    ├── audio_recorder.py    # Microphone capture & WAV serialization
    ├── transcription.py     # Local faster-whisper STT & QThread worker
    └── llm_client.py        # Local Ollama client & streaming QThread worker
```

---

## 🛡️ Privacy & Local Execution Guarantee

- **Zero Cloud APIs**: No OpenAI, Anthropic, Google Cloud, or third-party web endpoints are called.
- **Audio Privacy**: Audio is recorded exclusively into a local temporary folder (`temp_recordings/`) and automatically wiped upon transcription and application shutdown.
- **Inference Privacy**: Prompts are sent solely to localhost `http://127.0.0.1:11434`. Disconnecting from the internet entirely will not affect transcription or answer generation.

---

## ⚠️ Troubleshooting & Common Issues

### 1. "Ollama is not running. Please start Ollama desktop app or service."
- **Cause**: The Ollama background daemon is stopped.
- **Fix**: Open the Ollama app from the Windows Start menu or run `ollama serve` in a terminal. Then click the refresh button (`↻`) in the assistant header.

### 2. "Model 'qwen2.5:3b' not found in local Ollama library"
- **Cause**: The selected model has not been downloaded yet.
- **Fix**: Run `ollama pull qwen2.5:3b` (or `ollama pull llama3.2`) in your terminal, or select an installed model from the dropdown.

### 3. "No microphone found: Please connect an input device."
- **Cause**: Windows microphone permissions are disabled or no audio input device is detected.
- **Fix**: Open **Windows Settings > Privacy & security > Microphone** and ensure **Microphone access** and **Let desktop apps access your microphone** are turned **ON**.

### 4. "Recording too short"
- **Cause**: Stop was pressed within less than 0.25 seconds of starting.
- **Fix**: Speak for at least half a second before clicking Stop Recording.

---

## 📄 License
MIT License. Built for Internship Assignment.
