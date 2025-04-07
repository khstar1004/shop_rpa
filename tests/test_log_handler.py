"""Unit tests for the log_handler module"""

import unittest
import sys
import os
import logging
from unittest.mock import patch, MagicMock, Mock
import tempfile
import io

# Add project root to path for imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.log_handler import GUILogHandler, ColoredFormatter
from PyQt5.QtWidgets import QApplication, QPlainTextEdit
from PyQt5.QtGui import QColor, QTextCursor

class TestColoredFormatter(unittest.TestCase):
    """Test cases for the ColoredFormatter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.formatter = ColoredFormatter()
    
    def test_format_debug(self):
        """Test formatting of DEBUG level log record"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Debug message",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)
        self.assertIn("Debug message", result)
        self.assertIn("color:#", result)  # Should contain a color specification
    
    def test_format_info(self):
        """Test formatting of INFO level log record"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Info message",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)
        self.assertIn("Info message", result)
        self.assertIn("color:#", result)  # Should contain a color specification
    
    def test_format_warning(self):
        """Test formatting of WARNING level log record"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)
        self.assertIn("Warning message", result)
        self.assertIn("color:#", result)  # Should contain a color specification
    
    def test_format_error(self):
        """Test formatting of ERROR level log record"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)
        self.assertIn("Error message", result)
        self.assertIn("color:#", result)  # Should contain a color specification
    
    def test_format_critical(self):
        """Test formatting of CRITICAL level log record"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.CRITICAL,
            pathname="",
            lineno=0,
            msg="Critical message",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)
        self.assertIn("Critical message", result)
        self.assertIn("color:#", result)  # Should contain a color specification

class TestGUILogHandler(unittest.TestCase):
    """Test cases for the GUILogHandler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a QApplication instance
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication([])
        
        # Create a QPlainTextEdit instance
        self.text_edit = QPlainTextEdit()
        
        # Create a GUILogHandler instance
        self.handler = GUILogHandler(self.text_edit)
        
        # Add formatter to handler
        self.handler.setFormatter(logging.Formatter('%(message)s'))
    
    def test_emit(self):
        """Test emitting a log record"""
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test log message",
            args=(),
            exc_info=None
        )
        
        # Mock the appendHtml method to avoid gui updates
        with patch.object(self.text_edit, 'appendHtml') as mock_append:
            # Emit the record
            self.handler.emit(record)
            
            # Check that appendHtml was called
            mock_append.assert_called_once()
    
    def test_clear_log(self):
        """Test clearing the log widget"""
        # First add some content to the text edit
        self.text_edit.setPlainText("Some log content")
        
        # Mock the clear method to avoid gui updates
        with patch.object(self.text_edit, 'clear') as mock_clear:
            # Clear the log
            self.handler.clear_log()
            
            # Check that clear was called
            mock_clear.assert_called_once()
    
    def test_max_lines(self):
        """Test max lines functionality"""
        # Set a max lines limit
        max_lines = 3
        self.handler.set_max_lines(max_lines)
        
        # Add 5 log messages
        for i in range(5):
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Log message {i}",
                args=(),
                exc_info=None
            )
            
            with patch.object(self.text_edit, 'appendHtml'):
                self.handler.emit(record)
        
        # The text edit should have a maximum of max_lines lines
        with patch.object(self.text_edit, 'document') as mock_document:
            mock_doc = MagicMock()
            mock_doc.lineCount.return_value = 5
            mock_document.return_value = mock_doc
            
            with patch.object(self.text_edit, 'textCursor') as mock_cursor_method:
                mock_cursor = MagicMock()
                mock_cursor_method.return_value = mock_cursor
                
                with patch.object(self.text_edit, 'setTextCursor'):
                    # This would normally trim the text
                    self.handler.emit(record)
                    
                    # Check that methods to remove lines were called
                    self.assertTrue(mock_cursor.movePosition.called)
                    self.assertTrue(mock_cursor.movePosition.call_count >= 1)
    
    def test_save_log(self):
        """Test saving log to file"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Add content to the text edit
            test_content = "Test log content"
            self.text_edit.setPlainText(test_content)
            
            # Save the log
            self.handler.save_log(temp_path)
            
            # Read the file and check its content
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertEqual(content, test_content)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == '__main__':
    unittest.main()