"""Custom log handler for GUI logging"""

import logging
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QColor, QTextCursor

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color information to log records"""
    
    COLORS = {
        logging.DEBUG: QColor("#808080"),  # Gray
        logging.INFO: QColor("#000000"),   # Black
        logging.WARNING: QColor("#FFA500"), # Orange
        logging.ERROR: QColor("#FF0000"),   # Red
        logging.CRITICAL: QColor("#800000") # Dark Red
    }
    
    def __init__(self):
        super().__init__("%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
    def format(self, record):
        """Format the log record with color information"""
        formatted_text = super().format(record)
        return formatted_text

class GUILogHandler(logging.Handler, QObject):
    """Custom log handler that displays logs in a QPlainTextEdit widget"""
    
    def __init__(self, text_widget, max_lines=1000):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.text_widget = text_widget
        self.max_lines = max_lines
        self.formatter = ColoredFormatter()
        
    def emit(self, record):
        """Emit a log record to the text widget"""
        try:
            text = self.formatter.format(record)
            color = self.formatter.COLORS.get(record.levelno, QColor("#000000"))
            
            # Format the text with color
            html = f'<span style="color: {color.name()}">{text}</span><br>'
            self.text_widget.appendHtml(html)
            
            # Limit the number of lines
            doc = self.text_widget.document()
            while doc.lineCount() > self.max_lines:
                cursor = self.text_widget.textCursor()
                cursor.movePosition(QTextCursor.Start)
                cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 1)
                cursor.removeSelectedText()
                cursor.deletePreviousChar()  # Remove the newline
                
            self.text_widget.ensureCursorVisible()
            
        except Exception as e:
            self.handleError(record)
            
    def clear_log(self):
        """Clear all logs from the text widget"""
        self.text_widget.clear()
        
    def save_log(self, filename):
        """Save the current log to a file"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.text_widget.toPlainText())
            return True
        except Exception as e:
            logging.error(f"Error saving log to {filename}: {str(e)}")
            return False
            
    def set_max_lines(self, max_lines):
        """Set the maximum number of lines to keep in the log"""
        self.max_lines = max_lines
        # Trim existing log if necessary
        doc = self.text_widget.document()
        while doc.lineCount() > self.max_lines:
            cursor = self.text_widget.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 1)
            cursor.removeSelectedText()
            cursor.deletePreviousChar() 