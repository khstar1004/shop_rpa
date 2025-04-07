"""GUI widget initialization and configuration module"""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QProgressBar, QPlainTextEdit, QLabel,
    QTabWidget, QWidget, QStatusBar, QCheckBox
)
from PyQt5.QtCore import Qt, QByteArray
from PyQt5.QtSvg import QSvgWidget
from .styles import Styles
import os
from typing import Tuple, Optional
from PyQt5.QtGui import QIcon
from .i18n import translator as tr
from PyQt5.QtCore import pyqtSignal

class DropArea(QLabel):
    """Custom drop area widget for file drag and drop"""
    
    clicked = pyqtSignal()  # Add clicked signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        
        # Create SVG widget for upload icon
        self.svg_widget = QSvgWidget("gui/assets/file-upload.svg")
        self.svg_widget.setFixedSize(64, 64)
        
        # Create layout to center the SVG and text
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.svg_widget, 0, Qt.AlignCenter)
        
        # Add spacer
        layout.addSpacing(10)
        
        # Add text label
        self.text_label = QLabel(tr.get_text("file_drag_hint"))
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)
        
        # Setting properties
        self.setAcceptDrops(True)
        self.setMinimumHeight(180)
        self.is_dark_mode = False
        self.update_style()
    
    def update_style(self):
        """Update style based on dark mode setting"""
        if self.is_dark_mode:
            Styles.apply_drop_area_dark_style(self)
        else:
            Styles.apply_drop_area_style(self)
    
    def set_dark_mode(self, is_dark):
        """Set dark mode"""
        self.is_dark_mode = is_dark
        self.update_style()
    
    def highlight_active(self):
        """Highlight the drop area when file is being dragged over"""
        if self.is_dark_mode:
            Styles.apply_drop_area_active_dark_style(self)
        else:
            Styles.apply_drop_area_active_style(self)
        
        # Change SVG color to match
        self.svg_widget.load(QByteArray(
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            b'stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            b'<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
            b'<polyline points="17 8 12 3 7 8"></polyline>'
            b'<line x1="12" y1="3" x2="12" y2="15"></line>'
            b'</svg>'
        ))
    
    def highlight_inactive(self):
        """Remove highlight from drop area"""
        self.update_style()
        
        # Reset SVG
        self.svg_widget.load("gui/assets/file-upload.svg")
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class WidgetFactory:
    """Factory class for creating and configuring GUI widgets"""
    
    @staticmethod
    def create_file_group() -> Tuple[QGroupBox, DropArea, QLabel]:
        """Create file selection group with drag & drop area"""
        group = QGroupBox(tr.get_text("file_selection"))
        Styles.apply_group_box_style(group)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Create drop area
        drop_area = DropArea()
        layout.addWidget(drop_area)
        
        # File path display
        file_path = QLabel(tr.get_text("no_file"))
        file_path.setAlignment(Qt.AlignCenter)
        Styles.apply_file_path_style(file_path)
        layout.addWidget(file_path)
        
        group.setLayout(layout)
        return group, drop_area, file_path
    
    @staticmethod
    def create_controls_group() -> Tuple[QGroupBox, QPushButton, QPushButton, QSpinBox]:
        """Create processing controls group"""
        group = QGroupBox(tr.get_text("processing_control"))
        Styles.apply_group_box_style(group)
        
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        # Thread count control
        thread_layout = QHBoxLayout()
        thread_label = QLabel(tr.get_text("thread_count"))
        threads_spin = QSpinBox()
        threads_spin.setRange(1, os.cpu_count() or 1)
        Styles.apply_spinbox_style(threads_spin)
        thread_layout.addWidget(thread_label)
        thread_layout.addWidget(threads_spin)
        layout.addLayout(thread_layout)
        
        # Start button
        start_button = QPushButton(tr.get_text("start"))
        start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        start_button.setEnabled(False)
        Styles.apply_start_button_style(start_button)
        layout.addWidget(start_button)
        
        # Stop button
        stop_button = QPushButton(tr.get_text("stop"))
        stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        stop_button.setEnabled(False)
        Styles.apply_stop_button_style(stop_button)
        layout.addWidget(stop_button)
        
        group.setLayout(layout)
        return group, start_button, stop_button, threads_spin
    
    @staticmethod
    def create_progress_group() -> Tuple[QGroupBox, QProgressBar, QLabel]:
        """Create progress display group"""
        group = QGroupBox(tr.get_text("progress"))
        Styles.apply_group_box_style(group)
        
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setTextVisible(True)
        progress_bar.setValue(0)
        Styles.apply_progress_bar_style(progress_bar)
        layout.addWidget(progress_bar)
        
        # Status label
        status_label = QLabel(tr.get_text("waiting"))
        status_label.setAlignment(Qt.AlignCenter)
        Styles.apply_status_label_style(status_label)
        layout.addWidget(status_label)
        
        group.setLayout(layout)
        return group, progress_bar, status_label
    
    @staticmethod
    def create_log_area() -> QPlainTextEdit:
        """Create log display area"""
        log_area = QPlainTextEdit()
        log_area.setReadOnly(True)
        log_area.setMinimumHeight(150)
        Styles.apply_log_area_style(log_area)
        return log_area
    
    @staticmethod
    def create_settings_tab() -> Tuple[QWidget, QCheckBox, QCheckBox]:
        """Create settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # General settings group
        general_group = QGroupBox(tr.get_text("general_settings"))
        Styles.apply_group_box_style(general_group)
        general_layout = QVBoxLayout()
        
        # Dark mode toggle
        dark_mode_check = QCheckBox(tr.get_text("dark_mode"))
        Styles.apply_checkbox_style(dark_mode_check)
        general_layout.addWidget(dark_mode_check)
        
        # Auto-save toggle
        auto_save_check = QCheckBox(tr.get_text("auto_save"))
        Styles.apply_checkbox_style(auto_save_check)
        general_layout.addWidget(auto_save_check)
        
        general_group.setLayout(general_layout)
        layout.addWidget(general_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return tab, dark_mode_check, auto_save_check
    
    @staticmethod
    def create_status_bar() -> Tuple[QStatusBar, QLabel]:
        """Create status bar with memory usage display"""
        status_bar = QStatusBar()
        Styles.apply_status_bar_style(status_bar)
        
        memory_label = QLabel()
        Styles.apply_memory_label_style(memory_label)
        status_bar.addPermanentWidget(memory_label)
        
        return status_bar, memory_label
    
    @staticmethod
    def create_logo_widget() -> QSvgWidget:
        """Create and configure logo widget"""
        logo = QSvgWidget("gui/assets/logo.svg")
        logo.setFixedSize(48, 48)
        return logo 