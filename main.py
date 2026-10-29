"""
Local AI Assistant - Entry Point.
A privacy-first, fully local desktop application combining speech recognition
(faster-whisper) and local LLM inference (Ollama).
"""

import sys
import os

# Ensure the root directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow


def main():
    # Enable high-DPI scaling attributes if needed
    app = QApplication(sys.argv)
    app.setApplicationName("Local AI Assistant")
    app.setOrganizationName("LocalAI")

    # Smoke test support for automated verification without blocking
    if "--smoke-test" in sys.argv:
        print("[SMOKE TEST] Initializing Local AI Assistant MainWindow...")
        window = MainWindow()
        window.show()
        print("[SMOKE TEST] MainWindow initialized successfully. Closing.")
        window.close()
        return 0

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
