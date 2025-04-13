"""File manager widget"""

import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog
)

class FileManager(QWidget):
    """File manager widget for managing recent and favorite files"""
    
    file_selected = pyqtSignal(str)  # Signal emitted when a file is selected
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.initUI()
    
    def initUI(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Recent files section
        recent_group = QWidget()
        recent_layout = QVBoxLayout(recent_group)
        
        recent_header = QHBoxLayout()
        recent_label = QLabel("Recent Files")
        recent_label.setStyleSheet("font-weight: bold;")
        clear_recent = QPushButton("Clear")
        clear_recent.clicked.connect(self.on_clear_recent)
        recent_header.addWidget(recent_label)
        recent_header.addStretch()
        recent_header.addWidget(clear_recent)
        
        self.recent_list = QListWidget()
        self.recent_list.itemClicked.connect(self.on_file_selected)
        self.update_recent_files()
        
        recent_layout.addLayout(recent_header)
        recent_layout.addWidget(self.recent_list)
        
        # Favorite files section
        favorite_group = QWidget()
        favorite_layout = QVBoxLayout(favorite_group)
        
        favorite_header = QHBoxLayout()
        favorite_label = QLabel("Favorite Files")
        favorite_label.setStyleSheet("font-weight: bold;")
        favorite_header.addWidget(favorite_label)
        favorite_header.addStretch()
        
        self.favorite_list = QListWidget()
        self.favorite_list.itemClicked.connect(self.on_file_selected)
        self.update_favorite_files()
        
        favorite_layout.addLayout(favorite_header)
        favorite_layout.addWidget(self.favorite_list)
        
        # Add to main layout
        layout.addWidget(recent_group)
        layout.addWidget(favorite_group)
        layout.addStretch()
    
    def update_recent_files(self):
        """Update recent files list"""
        self.recent_list.clear()
        recent_files = self.settings.get("recent_files", [])
        
        for file_path in recent_files:
            if os.path.exists(file_path):
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                item.setToolTip(file_path)
                self.recent_list.addItem(item)
    
    def update_favorite_files(self):
        """Update favorite files list"""
        self.favorite_list.clear()
        favorite_files = self.settings.get("favorite_files", [])
        
        for file_path in favorite_files:
            if os.path.exists(file_path):
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                item.setToolTip(file_path)
                self.favorite_list.addItem(item)
    
    def on_file_selected(self, item):
        """Handle file selection"""
        file_path = item.data(Qt.UserRole)
        if os.path.exists(file_path):
            self.file_selected.emit(file_path)
        else:
            QMessageBox.warning(self, "Error", "File not found.")
            self.update_recent_files()
            self.update_favorite_files()
    
    def on_open_recent(self):
        """Open recent file"""
        item = self.recent_list.currentItem()
        if item:
            self.on_file_selected(item)
    
    def on_add_to_favorites(self):
        """Add current file to favorites"""
        item = self.recent_list.currentItem()
        if item:
            file_path = item.data(Qt.UserRole)
            favorite_files = self.settings.get("favorite_files", [])
            
            if file_path not in favorite_files:
                favorite_files.append(file_path)
                self.settings.set("favorite_files", favorite_files)
                self.update_favorite_files()
    
    def on_clear_recent(self):
        """Clear recent files list"""
        self.settings.set("recent_files", [])
        self.update_recent_files()
    
    def on_open_favorite(self):
        """Open favorite file"""
        item = self.favorite_list.currentItem()
        if item:
            self.on_file_selected(item)
    
    def on_remove_favorite(self):
        """Remove file from favorites"""
        item = self.favorite_list.currentItem()
        if item:
            file_path = item.data(Qt.UserRole)
            favorite_files = self.settings.get("favorite_files", [])
            
            if file_path in favorite_files:
                favorite_files.remove(file_path)
                self.settings.set("favorite_files", favorite_files)
                self.update_favorite_files()
    
    def add_to_recent_files(self, file_path):
        """Add file to recent files list"""
        recent_files = self.settings.get("recent_files", [])
        
        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add to beginning
        recent_files.insert(0, file_path)
        
        # Limit to 10 files
        if len(recent_files) > 10:
            recent_files = recent_files[:10]
        
        self.settings.set("recent_files", recent_files)
        self.update_recent_files() 