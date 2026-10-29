"""
Audio recording module using sounddevice and soundfile.
Handles microphone device discovery, streaming capture, RMS level tracking,
and saving to clean temporary WAV files for speech-to-text.
"""

import os
import queue
import threading
import time
import uuid
from typing import Callable, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """Thread-safe microphone recorder tailored for local speech-to-text."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        temp_dir: Optional[str] = None,
        level_callback: Optional[Callable[[float], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.temp_dir = temp_dir or os.path.join(os.getcwd(), "temp_recordings")
        os.makedirs(self.temp_dir, exist_ok=True)

        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_recording = False
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self.level_callback = level_callback
        self._recorded_frames: List[np.ndarray] = []

    @property
    def is_recording(self) -> bool:
        """Returns True if recording is currently active."""
        with self._lock:
            return self._is_recording

    @property
    def elapsed_time(self) -> float:
        """Returns elapsed recording duration in seconds."""
        if not self._is_recording or self._start_time == 0.0:
            return 0.0
        return time.time() - self._start_time

    @staticmethod
    def get_input_devices() -> List[Tuple[int, str]]:
        """List all available audio input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Device #{idx}")
                    devices.append((idx, name))
        except Exception:
            pass
        return devices

    @staticmethod
    def check_default_input_device() -> Tuple[bool, str]:
        """Check if a default input device is configured and available."""
        try:
            default_device = sd.query_devices(kind="input")
            if default_device and default_device.get("max_input_channels", 0) > 0:
                return True, default_device.get("name", "Default Input")
            return False, "No microphone input channels detected."
        except Exception as exc:
            return False, f"Audio device error: {exc}"

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback invoked by sounddevice for each audio block."""
        if status:
            pass  # Overflow or underflow warning

        if not self._is_recording:
            return

        # Make copy of array data
        data_copy = indata.copy()
        self._audio_queue.put(data_copy)

        # Calculate RMS amplitude for real-time visualizer
        if self.level_callback:
            try:
                rms = np.sqrt(np.mean(np.square(data_copy)))
                # Normalize approx to 0.0 - 1.0 range
                normalized_level = min(1.0, float(rms * 5.0))
                self.level_callback(normalized_level)
            except Exception:
                pass

    def start_recording(self, device_idx: Optional[int] = None) -> None:
        """Start capturing audio from the specified or default microphone."""
        with self._lock:
            if self._is_recording:
                raise RuntimeError("Recording is already in progress.")

            # Validate input device existence
            has_device, device_msg = self.check_default_input_device()
            if not has_device and device_idx is None:
                raise RuntimeError(
                    f"No microphone found: {device_msg}. Please connect an input device."
                )

            # Reset state
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            self._recorded_frames = []
            self._start_time = time.time()
            self._is_recording = True

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    callback=self._audio_callback,
                    device=device_idx,
                    blocksize=1024,
                )
                self._stream.start()
            except Exception as exc:
                self._is_recording = False
                self._stream = None
                raise RuntimeError(f"Failed to start audio stream: {exc}") from exc

    def stop_recording(self) -> str:
        """Stop capturing and save audio to a temporary WAV file.
        
        Returns:
            str: Path to the generated WAV file.
        """
        with self._lock:
            if not self._is_recording:
                raise RuntimeError("No active recording session to stop.")

            self._is_recording = False

            # Stop and close the stream
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

        # Drain the queue
        frames = []
        while not self._audio_queue.empty():
            try:
                frames.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break

        if not frames:
            raise ValueError("No audio data was captured. Please speak into the microphone.")

        audio_data = np.concatenate(frames, axis=0)

        # Validate minimum duration (e.g. at least 0.25 seconds)
        duration = len(audio_data) / self.sample_rate
        if duration < 0.25:
            raise ValueError(
                f"Recording too short ({duration:.2f}s). Please record a longer speech segment."
            )

        # Generate unique temporary WAV filepath
        filename = f"rec_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        output_filepath = os.path.join(self.temp_dir, filename)

        try:
            sf.write(output_filepath, audio_data, self.sample_rate, subtype="PCM_16")
        except Exception as exc:
            raise IOError(f"Failed to save recorded audio file: {exc}") from exc

        return output_filepath

    def cleanup_temp_file(self, filepath: str) -> None:
        """Safely delete a single temporary audio file."""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

    def cleanup_old_recordings(self, max_age_seconds: int = 3600) -> None:
        """Remove leftover recording files older than specified seconds."""
        now = time.time()
        try:
            for fname in os.listdir(self.temp_dir):
                if fname.endswith(".wav"):
                    fpath = os.path.join(self.temp_dir, fname)
                    if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_seconds:
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass
        except Exception:
            pass
