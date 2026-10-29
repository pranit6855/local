"""
Local LLM Client for Ollama.
Streams inference token-by-token directly from the local Ollama instance (127.0.0.1:11434).
Guarantees 100% offline local inference with zero cloud APIs.
Includes PySide6 QThread worker for smooth, non-blocking token streaming.
"""

import json
from typing import Generator, List, Optional, Tuple

from PySide6.QtCore import QObject, QThread, Signal
import requests


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b"
FALLBACK_MODEL = "llama3.2:latest"

ASSISTANT_SYSTEM_PROMPT = (
    "You are an intelligent, articulate, and helpful desktop AI assistant. "
    "Provide clear, direct, and well-structured answers to the user's spoken questions or prompts. "
    "Use bullet points or concise paragraphs where helpful. Keep explanations focused and natural."
)


class OllamaClient:
    """Interface to communicate with the local Ollama daemon."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST):
        self.host = host.rstrip("/")

    def check_connection(self) -> Tuple[bool, str, List[str]]:
        """Verify if Ollama is running and fetch all locally downloaded models.

        Returns:
            Tuple: (is_online, status_message, list_of_available_models)
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                if models:
                    return True, f"Ollama Online ({len(models)} model{'s' if len(models) > 1 else ''} found)", models
                return True, "Ollama Online (No models downloaded yet)", []
            return False, f"Ollama returned HTTP status {resp.status_code}", []
        except requests.exceptions.ConnectionError:
            return (
                False,
                "Ollama is not running. Please start the Ollama desktop app or service.",
                [],
            )
        except requests.exceptions.Timeout:
            return False, "Connection to local Ollama timed out.", []
        except Exception as exc:
            return False, f"Ollama connection error: {exc}", []

    def stream_generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        system_prompt: str = ASSISTANT_SYSTEM_PROMPT,
        stop_flag_callback: Optional[callable] = None,
    ) -> Generator[str, None, None]:
        """Stream generated text chunks from local Ollama.

        Yields:
            str: Each incoming token / word chunk.
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=(5.0, 60.0)) as response:
                if response.status_code == 404:
                    raise RuntimeError(
                        f"Model '{model}' not found in local Ollama library.\n"
                        f"Download it by running: `ollama run {model}` in your terminal."
                    )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Ollama error (HTTP {response.status_code}): {response.text}"
                    )

                for line in response.iter_lines(decode_unicode=True):
                    if stop_flag_callback and stop_flag_callback():
                        break

                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token

                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Lost connection to local Ollama. Ensure the Ollama server is running locally."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request error: {exc}") from exc


class LlmStreamWorker(QThread):
    """PySide6 QThread to stream LLM generation asynchronously into the UI."""

    started_generation = Signal()
    chunk_received = Signal(str)
    finished_generation = Signal(str)
    error_generation = Signal(str)

    def __init__(
        self,
        prompt: str,
        model: str,
        host: str = DEFAULT_OLLAMA_HOST,
        system_prompt: str = ASSISTANT_SYSTEM_PROMPT,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.prompt = prompt
        self.model = model
        self.host = host
        self.system_prompt = system_prompt
        self._is_cancelled = False
        self._client = OllamaClient(host=self.host)

    def cancel(self):
        """Signal cancellation to immediately break the stream."""
        self._is_cancelled = True

    def run(self):
        self.started_generation.emit()
        accumulated_text = []

        try:
            generator = self._client.stream_generate(
                prompt=self.prompt,
                model=self.model,
                system_prompt=self.system_prompt,
                stop_flag_callback=lambda: self._is_cancelled,
            )

            for token in generator:
                if self._is_cancelled:
                    break
                accumulated_text.append(token)
                self.chunk_received.emit(token)

            if not self._is_cancelled:
                full_text = "".join(accumulated_text)
                self.finished_generation.emit(full_text)

        except Exception as exc:
            self.error_generation.emit(str(exc))
