"""
Smoke test suite for Local AI Desktop Assistant.
Validates dependencies, audio device detection, Ollama connection, and GUI initialization.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    print("[TEST 1/5] Testing module imports...")
    try:
        import PySide6
        import sounddevice
        import soundfile
        import numpy
        import faster_whisper
        import requests
        print(" -> All core libraries imported successfully!")
        return True
    except ImportError as e:
        print(f" -> Import Error: {e}")
        return False


def test_audio_device():
    print("\n[TEST 2/5] Testing audio input devices...")
    try:
        from app.audio_recorder import AudioRecorder
        devices = AudioRecorder.get_input_devices()
        has_mic, mic_name = AudioRecorder.check_default_input_device()
        print(f" -> Found {len(devices)} input device(s).")
        print(f" -> Default mic status: {'Available' if has_mic else 'Unavailable'} ({mic_name})")
        return True
    except Exception as e:
        print(f" -> Audio Device check warning/error: {e}")
        return False


def test_ollama_connection():
    print("\n[TEST 3/5] Testing local Ollama connectivity...")
    try:
        from app.llm_client import OllamaClient
        client = OllamaClient()
        is_online, status_msg, models = client.check_connection()
        print(f" -> Ollama Online: {is_online}")
        print(f" -> Message: {status_msg}")
        print(f" -> Local models: {models}")
        return is_online, models
    except Exception as e:
        print(f" -> Ollama connection error: {e}")
        return False, []


def test_ollama_streaming(model_name: str):
    print(f"\n[TEST 4/5] Testing token streaming from local Ollama ({model_name})...")
    try:
        from app.llm_client import OllamaClient
        client = OllamaClient()
        prompt = "Reply with exactly three words: Local AI works."
        print(f" -> Prompt: '{prompt}'")
        tokens = []
        for token in client.stream_generate(prompt=prompt, model=model_name):
            tokens.append(token)
            print(token, end="", flush=True)
        print("\n -> Streaming generation successful!")
        return True
    except Exception as e:
        print(f"\n -> Ollama streaming error: {e}")
        return False


def test_gui_instantiation():
    print("\n[TEST 5/5] Testing PySide6 GUI window instantiation...")
    try:
        from PySide6.QtWidgets import QApplication
        from app.ui import MainWindow

        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        window = MainWindow()
        print(f" -> Window title: '{window.windowTitle()}'")
        print(f" -> Window size: {window.size().width()}x{window.size().height()}")
        window.close()
        print(" -> GUI instantiation passed!")
        return True
    except Exception as e:
        print(f" -> GUI initialization error: {e}")
        return False


def main():
    print("=" * 60)
    print("LOCAL AI DESKTOP ASSISTANT - SMOKE TESTS")
    print("=" * 60)

    t1 = test_imports()
    t2 = test_audio_device()
    t3, models = test_ollama_connection()

    t4 = False
    if t3 and models:
        chosen_model = models[0]
        t4 = test_ollama_streaming(chosen_model)
    else:
        print("\n[TEST 4/5] Skipped Ollama streaming test (Ollama not running or no models installed).")

    t5 = test_gui_instantiation()

    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"  Imports:            {'PASS' if t1 else 'FAIL'}")
    print(f"  Audio Check:        {'PASS' if t2 else 'FAIL'}")
    print(f"  Ollama Connection:  {'PASS' if t3 else 'WARN/OFFLINE'}")
    print(f"  Ollama Streaming:   {'PASS' if t4 else 'SKIPPED/FAIL'}")
    print(f"  GUI Instantiation:  {'PASS' if t5 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
