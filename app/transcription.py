"""
Local Speech-to-Text module using faster-whisper and soundfile.
Runs 100% offline on CPU with int8 quantization for high speed and low memory footprint.
Decodes audio directly using soundfile + numpy to bypass any external PyAV/FFmpeg version issues.
Includes both core transcriber and PySide6 QThread worker for seamless UI integration.
"""

import os
import sys
import threading
from typing import Callable, Optional, Tuple

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QThread, Signal


class WhisperTranscriber:
    """Core local Whisper transcription manager with model caching."""

    _model_instance = None
    _cached_model_size = None
    _lock = threading.Lock()

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def get_model(self, status_callback: Optional[Callable[[str], None]] = None):
        """Retrieve or initialize the faster-whisper model."""
        with self._lock:
            if (
                WhisperTranscriber._model_instance is not None
                and WhisperTranscriber._cached_model_size == self.model_size
            ):
                return WhisperTranscriber._model_instance

            if status_callback:
                status_callback(f"Loading local Whisper ({self.model_size}) model...")

            # Import faster_whisper lazily to avoid startup delay
            from faster_whisper import WhisperModel

            # Initialize Whisper on CPU with int8 quantization for maximum speed & compatibility
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
            )

            WhisperTranscriber._model_instance = model
            WhisperTranscriber._cached_model_size = self.model_size
            return model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, dict]:
        """Transcribe an audio file locally without sending data to external servers.

        Args:
            audio_path: Path to the WAV file.
            language: Language code (e.g. 'en') or None for auto-detection.
            status_callback: Optional callback for status messages.

        Returns:
            Tuple of (transcribed_text, info_dict)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = os.path.getsize(audio_path)
        if file_size < 1000:
            raise ValueError("Audio file is too small or contains no recorded speech.")

        # Read audio waveform directly using soundfile for maximum reliability
        try:
            audio_data, sr = sf.read(audio_path, dtype="float32")
        except Exception as exc:
            raise IOError(f"Failed to read audio file '{audio_path}': {exc}") from exc

        # Convert stereo to mono if necessary
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16,000 Hz if recorded at different sample rate
        target_sr = 16000
        if sr != target_sr and len(audio_data) > 0:
            target_len = int(len(audio_data) * target_sr / sr)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_len, endpoint=False),
                np.arange(len(audio_data)),
                audio_data,
            ).astype(np.float32)

        model = self.get_model(status_callback=status_callback)

        if status_callback:
            status_callback("Transcribing speech locally...")

        segments, info = model.transcribe(
            audio_data,
            beam_size=5,
            language=language,
            condition_on_previous_text=False,
            vad_filter=True,  # Voice activity detector to ignore silence
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        text_pieces = []
        for segment in segments:
            text_pieces.append(segment.text.strip())

        full_transcription = " ".join(text_pieces).strip()

        info_dict = {
            "language": info.language,
            "language_probability": getattr(info, "language_probability", 1.0),
            "duration": getattr(info, "duration", 0.0),
        }

        return full_transcription, info_dict


class TranscriptionWorker(QThread):
    """PySide6 QThread worker to execute transcription asynchronously without blocking UI."""

    progress = Signal(str)
    finished = Signal(str, dict)  # (transcribed_text, info_dict)
    error = Signal(str)

    def __init__(
        self,
        audio_path: str,
        model_size: str = "base",
        language: Optional[str] = None,
        cleanup_file: bool = False,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language
        self.cleanup_file = cleanup_file
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit("Preparing local Whisper engine...")
            transcriber = WhisperTranscriber(model_size=self.model_size)

            def on_status(msg: str):
                if not self._is_cancelled:
                    self.progress.emit(msg)

            text, info = transcriber.transcribe(
                self.audio_path,
                language=self.language,
                status_callback=on_status,
            )

            if self._is_cancelled:
                return

            self.finished.emit(text, info)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if self.cleanup_file and os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                except Exception:
                    pass
