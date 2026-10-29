"""
Whisper engine verification test.
Tests downloading and initializing the faster-whisper model on CPU int8,
and transcribing a short synthesized audio wave.
"""

import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.transcription import WhisperTranscriber


def test_whisper_engine():
    print("[WHISPER TEST] Initializing WhisperTranscriber (base, CPU int8)...")
    transcriber = WhisperTranscriber(model_size="base", device="cpu", compute_type="int8")

    # Generate a temporary 1-second 16kHz sine wave to verify transcription pipeline
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate 440 Hz tone
    audio_data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    os.makedirs("temp_recordings", exist_ok=True)
    test_wav = os.path.join("temp_recordings", "test_synth.wav")
    sf.write(test_wav, audio_data, sample_rate, subtype="PCM_16")

    print(f"[WHISPER TEST] Created test WAV: {test_wav}")
    print("[WHISPER TEST] Loading model and running transcription...")

    try:
        text, info = transcriber.transcribe(
            test_wav,
            status_callback=lambda msg: print(f"  [Status] {msg}")
        )
        print(f"[WHISPER TEST] SUCCESS! Model transcribed without error.")
        print(f"  Result text: '{text}'")
        print(f"  Detected language: {info.get('language')}")
        return True
    except Exception as e:
        print(f"[WHISPER TEST] Failed: {e}")
        return False
    finally:
        if os.path.exists(test_wav):
            try:
                os.remove(test_wav)
            except Exception:
                pass


if __name__ == "__main__":
    success = test_whisper_engine()
    sys.exit(0 if success else 1)
