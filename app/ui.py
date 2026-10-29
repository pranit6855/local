"""
User Interface for Local AI Desktop Assistant.
Built with PySide6, featuring a modern, dark-themed, Parakeet-inspired desktop layout.
Non-blocking operations with QThread workers for recording, transcription, and streaming LLM inference.
"""

import os
import sys
import time
from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.audio_recorder import AudioRecorder
from app.llm_client import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    FALLBACK_MODEL,
    LlmStreamWorker,
    OllamaClient,
)
from app.transcription import TranscriptionWorker


# Dark Theme Stylesheet (Parakeet-inspired dark palette)
DARK_STYLESHEET = """
QMainWindow {
    background-color: #0b0f19;
}

QWidget {
    color: #e2e8f0;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    font-size: 13px;
}

/* Header and Card Containers */
#HeaderCard {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 12px 16px;
}

#ControlCard {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 14px 18px;
}

#PanelCard {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 14px;
}

/* App Title & Badges */
#AppTitle {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.3px;
}

#AppSubtitle {
    font-size: 11px;
    color: #94a3b8;
    font-weight: 400;
}

#StatusBadge {
    padding: 4px 10px;
    border-radius: 14px;
    font-size: 11px;
    font-weight: 600;
}

#StatusBadgeOnline {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}

#StatusBadgeOffline {
    background-color: #450a0a;
    color: #f87171;
    border: 1px solid #dc2626;
}

#LocalBadge {
    background-color: #1e1b4b;
    color: #a5b4fc;
    border: 1px solid #3730a3;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

/* Primary Action Buttons */
QPushButton {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}

QPushButton:pressed {
    background-color: #0f172a;
}

QPushButton:disabled {
    background-color: #1e293b;
    color: #475569;
    border-color: #1e293b;
}

/* Start Recording Button */
#BtnRecordStart {
    background-color: #dc2626;
    color: #ffffff;
    border: 1px solid #ef4444;
}

#BtnRecordStart:hover {
    background-color: #b91c1c;
}

#BtnRecordStart:disabled {
    background-color: #3f1d24;
    color: #7f3d49;
    border-color: #3f1d24;
}

/* Stop Recording Button */
#BtnRecordStop {
    background-color: #475569;
    color: #ffffff;
    border: 1px solid #64748b;
}

#BtnRecordStop:hover {
    background-color: #334155;
}

#BtnRecordStop:disabled {
    background-color: #1e293b;
    color: #475569;
    border-color: #1e293b;
}

/* Suggest Answer Primary Button */
#BtnSuggestAnswer {
    background-color: #4f46e5;
    color: #ffffff;
    border: 1px solid #6366f1;
    padding: 9px 20px;
    font-size: 13px;
    font-weight: 700;
}

#BtnSuggestAnswer:hover {
    background-color: #4338ca;
    border-color: #4f46e5;
}

#BtnSuggestAnswer:disabled {
    background-color: #242247;
    color: #555385;
    border-color: #242247;
}

/* Secondary Buttons */
#BtnSecondary {
    background-color: transparent;
    color: #94a3b8;
    border: 1px solid #334155;
    padding: 5px 10px;
    font-size: 11px;
    border-radius: 6px;
}

#BtnSecondary:hover {
    background-color: #1e293b;
    color: #f1f5f9;
}

/* Text Editors */
QTextEdit {
    background-color: #0d1321;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 12px;
    color: #f1f5f9;
    font-size: 13px;
    line-height: 1.5;
    selection-background-color: #3b82f6;
    selection-color: #ffffff;
}

QTextEdit:focus {
    border: 1px solid #3b82f6;
}

/* Combobox */
QComboBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 12px;
    color: #f1f5f9;
    min-width: 140px;
}

QComboBox:hover {
    border-color: #475569;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    selection-background-color: #3b82f6;
    color: #f1f5f9;
    padding: 4px;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #0b0f19;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class AudioVisualizerBar(QWidget):
    """Real-time horizontal audio meter bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(10)
        self.setMinimumWidth(100)
        self._level = 0.0

    def set_level(self, level: float):
        """Update audio level (0.0 to 1.0)."""
        self._level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background track
        painter.setBrush(QColor("#1f2937"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 4, 4)

        # Active level fill
        if self._level > 0.01:
            active_width = int(self.width() * self._level)
            # Gradient color based on level intensity
            if self._level > 0.8:
                color = QColor("#ef4444")
            elif self._level > 0.5:
                color = QColor("#eab308")
            else:
                color = QColor("#10b981")
            painter.setBrush(color)
            painter.drawRoundedRect(0, 0, active_width, self.height(), 4, 4)


class MainWindow(QMainWindow):
    """Main Application Window for Local AI Desktop Assistant."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local AI Assistant")
        self.resize(1000, 750)
        self.setMinimumSize(820, 600)
        self.setStyleSheet(DARK_STYLESHEET)

        # Core Components
        self.audio_recorder = AudioRecorder(level_callback=self._on_audio_level)
        self.ollama_client = OllamaClient()

        # State tracking
        self.transcription_worker: Optional[TranscriptionWorker] = None
        self.llm_worker: Optional[LlmStreamWorker] = None
        self.current_audio_file: Optional[str] = None
        self.available_models: list[str] = []

        # Timers
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(200)
        self.record_timer.timeout.connect(self._update_recording_timer)

        self.health_timer = QTimer(self)
        self.health_timer.setInterval(8000)  # Check Ollama every 8s
        self.health_timer.timeout.connect(self._check_ollama_status)

        self._setup_ui()
        self._refresh_ollama(initial=True)

        # Clean old temporary audio files on startup
        self.audio_recorder.cleanup_old_recordings()

    def _setup_ui(self):
        """Construct the UI layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Header Card
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(8, 6, 8, 6)

        # Title block
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        app_title = QLabel("Local AI Assistant")
        app_title.setObjectName("AppTitle")
        app_subtitle = QLabel("100% Offline • Local Speech-to-Text & Local LLM Inference")
        app_subtitle.setObjectName("AppSubtitle")
        title_block.addWidget(app_title)
        title_block.addWidget(app_subtitle)
        header_layout.addLayout(title_block)

        header_layout.addStretch()

        # Model Selector block
        model_block = QHBoxLayout()
        model_block.setSpacing(8)

        lbl_model = QLabel("Ollama Model:")
        lbl_model.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        model_block.addWidget(lbl_model)

        self.combo_model = QComboBox()
        self.combo_model.setToolTip("Select locally installed Ollama model")
        model_block.addWidget(self.combo_model)

        btn_refresh_models = QPushButton("↻")
        btn_refresh_models.setObjectName("BtnSecondary")
        btn_refresh_models.setFixedSize(30, 28)
        btn_refresh_models.setToolTip("Refresh local Ollama status and models")
        btn_refresh_models.clicked.connect(lambda: self._refresh_ollama(initial=False))
        model_block.addWidget(btn_refresh_models)

        # Ollama Status Badge
        self.lbl_ollama_badge = QLabel("Checking Ollama...")
        self.lbl_ollama_badge.setObjectName("StatusBadge")
        model_block.addWidget(self.lbl_ollama_badge)

        header_layout.addLayout(model_block)
        main_layout.addWidget(header_card)

        # 2. Controls & Recording Card
        control_card = QFrame()
        control_card.setObjectName("ControlCard")
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(12)

        # Record Start / Stop buttons
        self.btn_record_start = QPushButton("● Start Recording")
        self.btn_record_start.setObjectName("BtnRecordStart")
        self.btn_record_start.setCursor(Qt.PointingHandCursor)
        self.btn_record_start.clicked.connect(self._on_start_recording)
        control_layout.addWidget(self.btn_record_start)

        self.btn_record_stop = QPushButton("■ Stop Recording")
        self.btn_record_stop.setObjectName("BtnRecordStop")
        self.btn_record_stop.setCursor(Qt.PointingHandCursor)
        self.btn_record_stop.setEnabled(False)
        self.btn_record_stop.clicked.connect(self._on_stop_recording)
        control_layout.addWidget(self.btn_record_stop)

        # Audio visualizer & duration indicator
        self.lbl_record_timer = QLabel("00:00")
        self.lbl_record_timer.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: 700; color: #94a3b8;"
        )
        control_layout.addWidget(self.lbl_record_timer)

        self.visualizer_bar = AudioVisualizerBar()
        control_layout.addWidget(self.visualizer_bar)

        control_layout.addStretch()

        # Operational status label
        self.lbl_status = QLabel("Ready to record or enter question.")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        control_layout.addWidget(self.lbl_status)

        main_layout.addWidget(control_card)

        # 3. Main Splitter with Panels (Transcription Panel & AI Answer Panel)
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(8)

        # Panel A: Transcription Panel
        panel_transcription = QFrame()
        panel_transcription.setObjectName("PanelCard")
        layout_trans = QVBoxLayout(panel_transcription)
        layout_trans.setContentsMargins(10, 10, 10, 10)
        layout_trans.setSpacing(8)

        trans_header = QHBoxLayout()
        lbl_trans_title = QLabel("Transcribed Question / Prompt")
        lbl_trans_title.setStyleSheet("font-weight: 700; font-size: 14px; color: #f1f5f9;")
        trans_header.addWidget(lbl_trans_title)

        trans_header.addStretch()

        self.lbl_trans_count = QLabel("0 words")
        self.lbl_trans_count.setStyleSheet("color: #64748b; font-size: 11px;")
        trans_header.addWidget(self.lbl_trans_count)

        btn_copy_trans = QPushButton("Copy")
        btn_copy_trans.setObjectName("BtnSecondary")
        btn_copy_trans.setToolTip("Copy transcribed text to clipboard")
        btn_copy_trans.clicked.connect(self._copy_transcription)
        trans_header.addWidget(btn_copy_trans)

        layout_trans.addLayout(trans_header)

        self.text_transcription = QTextEdit()
        self.text_transcription.setPlaceholderText(
            "Transcribed speech will appear here after recording, or you can type directly..."
        )
        self.text_transcription.textChanged.connect(self._on_transcription_text_changed)
        layout_trans.addWidget(self.text_transcription)

        # Action bar under transcription
        trans_action_bar = QHBoxLayout()
        lbl_hint = QLabel("Tip: Click 'Suggest Answer' or press Ctrl+Return")
        lbl_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        trans_action_bar.addWidget(lbl_hint)

        trans_action_bar.addStretch()

        self.btn_suggest_answer = QPushButton("✨ Suggest Answer")
        self.btn_suggest_answer.setObjectName("BtnSuggestAnswer")
        self.btn_suggest_answer.setCursor(Qt.PointingHandCursor)
        self.btn_suggest_answer.setEnabled(False)
        self.btn_suggest_answer.clicked.connect(self._on_suggest_answer)
        trans_action_bar.addWidget(self.btn_suggest_answer)

        layout_trans.addLayout(trans_action_bar)
        splitter.addWidget(panel_transcription)

        # Panel B: AI Answer Panel
        panel_answer = QFrame()
        panel_answer.setObjectName("PanelCard")
        layout_answer = QVBoxLayout(panel_answer)
        layout_answer.setContentsMargins(10, 10, 10, 10)
        layout_answer.setSpacing(8)

        answer_header = QHBoxLayout()
        lbl_answer_title = QLabel("AI Suggested Answer")
        lbl_answer_title.setStyleSheet("font-weight: 700; font-size: 14px; color: #f1f5f9;")
        answer_header.addWidget(lbl_answer_title)

        self.lbl_answer_badge = QLabel("Local LLM")
        self.lbl_answer_badge.setObjectName("LocalBadge")
        answer_header.addWidget(self.lbl_answer_badge)

        answer_header.addStretch()

        self.btn_stop_generation = QPushButton("■ Stop")
        self.btn_stop_generation.setObjectName("BtnSecondary")
        self.btn_stop_generation.setToolTip("Stop streaming AI response")
        self.btn_stop_generation.setVisible(False)
        self.btn_stop_generation.clicked.connect(self._on_stop_generation)
        answer_header.addWidget(self.btn_stop_generation)

        btn_copy_answer = QPushButton("Copy Answer")
        btn_copy_answer.setObjectName("BtnSecondary")
        btn_copy_answer.setToolTip("Copy full AI answer to clipboard")
        btn_copy_answer.clicked.connect(self._copy_answer)
        answer_header.addWidget(btn_copy_answer)

        layout_answer.addLayout(answer_header)

        self.text_answer = QTextEdit()
        self.text_answer.setReadOnly(True)
        self.text_answer.setPlaceholderText(
            "AI-generated response will stream here token-by-token..."
        )
        layout_answer.addWidget(self.text_answer)

        splitter.addWidget(panel_answer)
        splitter.setSizes([260, 380])
        main_layout.addWidget(splitter, 1)

        # 4. Footer Bar
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(4, 2, 4, 2)

        lbl_privacy = QLabel("🔒 Privacy Notice: All audio and text stay on this machine.")
        lbl_privacy.setStyleSheet("color: #64748b; font-size: 11px;")
        footer_layout.addWidget(lbl_privacy)

        footer_layout.addStretch()

        btn_clear_all = QPushButton("Reset / Clear All")
        btn_clear_all.setObjectName("BtnSecondary")
        btn_clear_all.setToolTip("Clear both transcription and AI answer")
        btn_clear_all.clicked.connect(self._on_clear_all)
        footer_layout.addWidget(btn_clear_all)

        main_layout.addLayout(footer_layout)

        # Keyboard shortcuts
        suggest_action = QAction(self)
        suggest_action.setShortcut(QKeySequence("Ctrl+Return"))
        suggest_action.triggered.connect(self._on_suggest_answer)
        self.addAction(suggest_action)

        # Start health check timer
        self.health_timer.start()

    # -------------------------------------------------------------------------
    # Audio Level & Timer Updates
    # -------------------------------------------------------------------------
    def _on_audio_level(self, level: float):
        """Thread-safe update of visualizer bar."""
        # Using invokeMethod or direct call since PyQt/PySide can update properties safely or through timer
        QTimer.singleShot(0, lambda: self.visualizer_bar.set_level(level))

    def _update_recording_timer(self):
        """Update recording duration timer display."""
        elapsed = self.audio_recorder.elapsed_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.lbl_record_timer.setText(f"{mins:02d}:{secs:02d}")

    # -------------------------------------------------------------------------
    # Recording Controls
    # -------------------------------------------------------------------------
    def _on_start_recording(self):
        """Handle Start Recording button click."""
        try:
            self.lbl_status.setText("● Recording microphone audio...")
            self.lbl_status.setStyleSheet("color: #f87171; font-weight: 600; font-size: 12px;")
            self.lbl_record_timer.setStyleSheet(
                "font-family: monospace; font-size: 14px; font-weight: 700; color: #ef4444;"
            )

            self.audio_recorder.start_recording()

            self.btn_record_start.setEnabled(False)
            self.btn_record_stop.setEnabled(True)
            self.btn_suggest_answer.setEnabled(False)
            self.record_timer.start()

        except Exception as exc:
            self._show_error("Recording Error", str(exc))
            self._reset_recording_ui()

    def _on_stop_recording(self):
        """Handle Stop Recording button click and begin transcription."""
        self.record_timer.stop()
        self.visualizer_bar.set_level(0.0)

        self.btn_record_stop.setEnabled(False)
        self.lbl_status.setText("Processing audio...")
        self.lbl_status.setStyleSheet("color: #fbbf24; font-weight: 600; font-size: 12px;")

        try:
            wav_path = self.audio_recorder.stop_recording()
            self.current_audio_file = wav_path
            self._start_transcription(wav_path)
        except Exception as exc:
            self._show_error("Recording Captured Error", str(exc))
            self._reset_recording_ui()

    def _reset_recording_ui(self):
        """Reset recording buttons and timer to idle state."""
        self.btn_record_start.setEnabled(True)
        self.btn_record_stop.setEnabled(False)
        self.visualizer_bar.set_level(0.0)
        self.lbl_record_timer.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: 700; color: #94a3b8;"
        )
        self.lbl_status.setText("Ready.")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self._on_transcription_text_changed()

    # -------------------------------------------------------------------------
    # Speech-to-Text Transcription
    # -------------------------------------------------------------------------
    def _start_transcription(self, wav_path: str):
        """Launch asynchronous faster-whisper transcription worker."""
        self.btn_record_start.setEnabled(False)
        self.btn_suggest_answer.setEnabled(False)

        self.lbl_status.setText("Transcribing speech with Whisper (CPU int8)...")
        self.lbl_status.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 12px;")

        self.transcription_worker = TranscriptionWorker(
            audio_path=wav_path,
            model_size="base",
            cleanup_file=True,
        )
        self.transcription_worker.progress.connect(self._on_transcription_progress)
        self.transcription_worker.finished.connect(self._on_transcription_finished)
        self.transcription_worker.error.connect(self._on_transcription_error)
        self.transcription_worker.start()

    def _on_transcription_progress(self, msg: str):
        self.lbl_status.setText(msg)

    def _on_transcription_finished(self, text: str, info: dict):
        self.btn_record_start.setEnabled(True)
        self._reset_recording_ui()

        if text.strip():
            self.text_transcription.setPlainText(text.strip())
            lang = info.get("language", "auto")
            dur = info.get("duration", 0.0)
            self.lbl_status.setText(f"✓ Transcribed ({dur:.1f}s, lang: {lang})")
            self.lbl_status.setStyleSheet("color: #34d399; font-size: 12px;")
        else:
            self.lbl_status.setText("Transcription completed, but no speech detected.")
            self.lbl_status.setStyleSheet("color: #fbbf24; font-size: 12px;")

    def _on_transcription_error(self, error_msg: str):
        self.btn_record_start.setEnabled(True)
        self._reset_recording_ui()
        self._show_error("Transcription Error", f"Speech recognition failed: {error_msg}")

    # -------------------------------------------------------------------------
    # Local LLM Inference
    # -------------------------------------------------------------------------
    def _on_suggest_answer(self):
        """Send transcription text to local Ollama and stream tokens into answer panel."""
        prompt = self.text_transcription.toPlainText().strip()
        if not prompt:
            self._show_error("Empty Prompt", "Please record speech or enter a question first.")
            return

        selected_model = self.combo_model.currentText()
        if not selected_model:
            selected_model = DEFAULT_MODEL

        # Clear answer panel and show stop button
        self.text_answer.clear()
        self.btn_suggest_answer.setEnabled(False)
        self.btn_stop_generation.setVisible(True)

        self.lbl_status.setText(f"Generating answer using {selected_model}...")
        self.lbl_status.setStyleSheet("color: #818cf8; font-weight: 600; font-size: 12px;")

        self.llm_worker = LlmStreamWorker(
            prompt=prompt,
            model=selected_model,
        )
        self.llm_worker.started_generation.connect(self._on_llm_started)
        self.llm_worker.chunk_received.connect(self._on_llm_chunk)
        self.llm_worker.finished_generation.connect(self._on_llm_finished)
        self.llm_worker.error_generation.connect(self._on_llm_error)
        self.llm_worker.start()

    def _on_llm_started(self):
        self.lbl_status.setText("Streaming response from local LLM...")

    def _on_llm_chunk(self, chunk: str):
        """Append incoming streaming token and smoothly auto-scroll."""
        self.text_answer.insertPlainText(chunk)
        # Scroll to bottom
        scrollbar = self.text_answer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_llm_finished(self, full_text: str):
        self.btn_suggest_answer.setEnabled(True)
        self.btn_stop_generation.setVisible(False)
        self.lbl_status.setText("✓ Response complete.")
        self.lbl_status.setStyleSheet("color: #34d399; font-size: 12px;")

    def _on_llm_error(self, error_msg: str):
        self.btn_suggest_answer.setEnabled(True)
        self.btn_stop_generation.setVisible(False)
        self.lbl_status.setText("Generation error.")
        self.lbl_status.setStyleSheet("color: #ef4444; font-size: 12px;")
        self._show_error("LLM Generation Error", error_msg)

    def _on_stop_generation(self):
        """Cancel the streaming LLM worker immediately."""
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.cancel()
            self.btn_stop_generation.setVisible(False)
            self.btn_suggest_answer.setEnabled(True)
            self.lbl_status.setText("Generation stopped by user.")
            self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")

    # -------------------------------------------------------------------------
    # Ollama Health & Model Refresh
    # -------------------------------------------------------------------------
    def _check_ollama_status(self):
        """Periodic background status check for Ollama."""
        is_online, msg, models = self.ollama_client.check_connection()
        self._apply_ollama_status(is_online, msg, models, update_combo=False)

    def _refresh_ollama(self, initial: bool = False):
        """Explicitly refresh Ollama connection and populate model list."""
        is_online, msg, models = self.ollama_client.check_connection()
        self._apply_ollama_status(is_online, msg, models, update_combo=True)

        if not initial and not is_online:
            self._show_error("Ollama Unavailable", msg)

    def _apply_ollama_status(
        self, is_online: bool, msg: str, models: list[str], update_combo: bool = False
    ):
        """Update status badges and combo options."""
        if is_online:
            self.lbl_ollama_badge.setText("● Ollama Online")
            self.lbl_ollama_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; border: 1px solid #059669; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 11px;"
            )
            self.lbl_ollama_badge.setToolTip(msg)
        else:
            self.lbl_ollama_badge.setText("● Ollama Offline")
            self.lbl_ollama_badge.setStyleSheet(
                "background-color: #450a0a; color: #f87171; border: 1px solid #dc2626; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 11px;"
            )
            self.lbl_ollama_badge.setToolTip(msg)

        if update_combo or (models and self.combo_model.count() == 0):
            current_choice = self.combo_model.currentText()
            self.combo_model.clear()

            if models:
                self.available_models = models
                for m in models:
                    self.combo_model.addItem(m)

                # Prioritize DEFAULT_MODEL or FALLBACK_MODEL
                if current_choice in models:
                    self.combo_model.setCurrentText(current_choice)
                elif DEFAULT_MODEL in models:
                    self.combo_model.setCurrentText(DEFAULT_MODEL)
                elif FALLBACK_MODEL in models:
                    self.combo_model.setCurrentText(FALLBACK_MODEL)
                elif any("qwen" in m.lower() for m in models):
                    match = next(m for m in models if "qwen" in m.lower())
                    self.combo_model.setCurrentText(match)
                elif any("llama" in m.lower() for m in models):
                    match = next(m for m in models if "llama" in m.lower())
                    self.combo_model.setCurrentText(match)
            else:
                # Add default placeholder if no models discovered yet
                self.combo_model.addItem(DEFAULT_MODEL)
                self.combo_model.addItem(FALLBACK_MODEL)

    # -------------------------------------------------------------------------
    # Helper & UI Event Handlers
    # -------------------------------------------------------------------------
    def _on_transcription_text_changed(self):
        """Enable or disable Suggest Answer button based on text presence."""
        text = self.text_transcription.toPlainText().strip()
        words = len(text.split()) if text else 0
        self.lbl_trans_count.setText(f"{words} word{'s' if words != 1 else ''}")

        if not self.audio_recorder.is_recording:
            self.btn_suggest_answer.setEnabled(bool(text))

    def _copy_transcription(self):
        """Copy transcription to clipboard."""
        text = self.text_transcription.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.lbl_status.setText("Transcribed text copied to clipboard.")

    def _copy_answer(self):
        """Copy AI answer to clipboard."""
        text = self.text_answer.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.lbl_status.setText("AI answer copied to clipboard.")

    def _on_clear_all(self):
        """Reset text inputs and status."""
        self.text_transcription.clear()
        self.text_answer.clear()
        self.lbl_record_timer.setText("00:00")
        self.lbl_status.setText("Cleared. Ready for next session.")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")

    def _show_error(self, title: str, message: str):
        """Display an actionable error message dialog."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(f"<b>{title}</b>")
        msg_box.setInformativeText(message)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setStyleSheet(
            "QMessageBox { background-color: #111827; } "
            "QLabel { color: #f8fafc; font-size: 13px; } "
            "QPushButton { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 6px 14px; border-radius: 6px; }"
        )
        msg_box.exec()

    def closeEvent(self, event):
        """Clean up threads and timers on application close."""
        self.record_timer.stop()
        self.health_timer.stop()

        if self.audio_recorder.is_recording:
            try:
                self.audio_recorder.stop_recording()
            except Exception:
                pass

        if self.transcription_worker and self.transcription_worker.isRunning():
            self.transcription_worker.cancel()
            self.transcription_worker.wait(1000)

        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.cancel()
            self.llm_worker.wait(1000)

        event.accept()
