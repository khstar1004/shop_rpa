"""Custom log handler for GUI logging"""

import logging
from typing import Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QTextCursor
from PyQt5.QtWidgets import QPlainTextEdit


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color information to log records"""

    COLORS = {
        logging.DEBUG: QColor("#808080"),  # Gray
        logging.INFO: QColor("#000000"),  # Black
        logging.WARNING: QColor("#FFA500"),  # Orange
        logging.ERROR: QColor("#FF0000"),  # Red
        logging.CRITICAL: QColor("#800000"),  # Dark Red
    }

    def __init__(self):
        super().__init__(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record):
        """Format the log record with color information"""
        formatted_text = super().format(record)
        return formatted_text


class GUILogHandler(logging.Handler, QObject):
    """Custom log handler that displays logs in a QPlainTextEdit widget"""

    def __init__(self, text_widget: Optional[QPlainTextEdit] = None, max_lines: int = 1000, signal: Optional[pyqtSignal] = None):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.text_widget = text_widget
        self.max_lines = max_lines
        self.signal = signal
        self.formatter = ColoredFormatter()
        self.setFormatter(self.formatter)

    def emit(self, record):
        """Emit a log record to the text widget and signal if available"""
        try:
            formatted_text = self.formatter.format(record)
            
            # Update text widget if available
            if self.text_widget:
                color = self.formatter.COLORS.get(record.levelno, QColor("#000000"))
                html = f'<span style="color: {color.name()}">{formatted_text}</span><br>'
                self.text_widget.appendHtml(html)
                self._trim_log()

            # Emit signal if available
            if self.signal:
                self.signal.emit(formatted_text)

        except Exception as e:
            self.handleError(record)
            logging.error(f"Error in GUILogHandler.emit: {str(e)}")

    def _trim_log(self):
        """Trim the log to maintain maximum line count"""
        try:
            doc = self.text_widget.document()
            while doc.lineCount() > self.max_lines:
                cursor = self.text_widget.textCursor()
                cursor.movePosition(QTextCursor.Start)
                cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 1)
                cursor.removeSelectedText()
                cursor.deletePreviousChar()  # Remove the newline
            self.text_widget.ensureCursorVisible()
        except Exception as e:
            logging.error(f"Error trimming log: {str(e)}")

    def clear_log(self):
        """Clear all logs from the text widget"""
        if self.text_widget:
            self.text_widget.clear()

    def save_log(self, filename: str) -> bool:
        """Save the current log to a file"""
        try:
            if self.text_widget:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.text_widget.toPlainText())
                return True
            return False
        except Exception as e:
            logging.error(f"Error saving log to {filename}: {str(e)}")
            return False

    def set_max_lines(self, max_lines: int):
        """Set the maximum number of lines to keep in the log"""
        self.max_lines = max_lines
        if self.text_widget:
            self._trim_log()
