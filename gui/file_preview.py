"""File preview widget"""

import os
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

class FilePreviewWidget(QWidget):
    """Excel file preview widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Preview label
        self.preview_label = QLabel("File Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        
        # Preview text area
        self.preview_area = QPlainTextEdit()
        self.preview_area.setReadOnly(True)
        self.preview_area.setMaximumHeight(150)
        self.preview_area.setPlaceholderText("File contents will be displayed here")
        layout.addWidget(self.preview_area)
    
    def set_file(self, file_path):
        """Set file to preview"""
        try:
            if file_path and file_path.lower().endswith(('.xlsx', '.xls')):
                # Load Excel file
                df = pd.read_excel(file_path, nrows=10)  # Load only first 10 rows
                
                # Display file preview
                self.preview_label.setText(f"File Preview: {os.path.basename(file_path)}")
                
                # Convert dataframe to text
                preview_text = df.to_string(index=False)
                self.preview_area.setPlainText(preview_text)
            else:
                self.preview_label.setText("Unsupported file format")
                self.preview_area.setPlainText("")
        except Exception as e:
            self.preview_label.setText("Preview failed")
            self.preview_area.setPlainText(f"Error: {str(e)}") 