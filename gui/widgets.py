"""GUI widget initialization and configuration module"""

import os
import logging
from typing import Tuple, Optional, Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .i18n import translator as tr
from .styles import Styles
from .drop_area import DropArea
from .progress_indicator import ProgressStepIndicator
from .resource_graph import ResourceUsageGraph
from .settings_tab import SettingsTabWidget
from .file_manager import FileManager
from .help_system import HelpSystem
from .performance_monitor import PerformanceMonitorWidget
from .toast_notification import ToastNotification

# Configure logging
logger = logging.getLogger(__name__)

class WidgetFactory:
    """Factory class for creating widgets"""
    
    @staticmethod
    def create_file_group() -> Tuple[QGroupBox, DropArea, QLabel]:
        """Create file selection group"""
        try:
            file_group = QGroupBox(tr.get_text("file_selection", "File Selection"))
            layout = QVBoxLayout(file_group)
            
            drop_area = DropArea()
            file_label = QLabel(tr.get_text("no_file_selected", "No file selected"))
            file_label.setAlignment(Qt.AlignCenter)
            
            layout.addWidget(drop_area)
            layout.addWidget(file_label)
            
            return file_group, drop_area, file_label
        except Exception as e:
            logger.error(f"Error creating file group: {str(e)}")
            raise
    
    @staticmethod
    def create_controls_group() -> Tuple[QGroupBox, QPushButton, QPushButton, QSpinBox]:
        """Create controls group"""
        try:
            controls_group = QGroupBox(tr.get_text("controls", "Controls"))
            layout = QVBoxLayout(controls_group)
            
            process_button = QPushButton(tr.get_text("process", "Process"))
            stop_button = QPushButton(tr.get_text("stop", "Stop"))
            thread_spin = QSpinBox()
            thread_spin.setRange(1, 8)
            thread_spin.setValue(4)
            thread_spin.setPrefix(tr.get_text("threads_prefix", "Threads: "))
            
            layout.addWidget(process_button)
            layout.addWidget(stop_button)
            layout.addWidget(thread_spin)
            
            return controls_group, process_button, stop_button, thread_spin
        except Exception as e:
            logger.error(f"Error creating controls group: {str(e)}")
            raise
    
    @staticmethod
    def create_progress_group() -> Tuple[QGroupBox, QProgressBar, QLabel, ProgressStepIndicator, ResourceUsageGraph]:
        """Create progress group"""
        try:
            progress_group = QGroupBox(tr.get_text("progress", "Progress"))
            layout = QVBoxLayout(progress_group)
            
            progress_bar = QProgressBar()
            status_label = QLabel(tr.get_text("status_ready", "Ready"))
            step_indicator = ProgressStepIndicator([
                tr.get_text("step_load", "Load"), 
                tr.get_text("step_process", "Process"), 
                tr.get_text("step_save", "Save")
            ])
            resource_graph = ResourceUsageGraph()
            
            layout.addWidget(progress_bar)
            layout.addWidget(status_label)
            layout.addWidget(step_indicator)
            layout.addWidget(resource_graph)
            
            return progress_group, progress_bar, status_label, step_indicator, resource_graph
        except Exception as e:
            logger.error(f"Error creating progress group: {str(e)}")
            raise
    
    @staticmethod
    def create_log_area() -> QPlainTextEdit:
        """Create log area"""
        try:
            log_area = QPlainTextEdit()
            log_area.setReadOnly(True)
            log_area.setMaximumHeight(150)
            return log_area
        except Exception as e:
            logger.error(f"Error creating log area: {str(e)}")
            raise
    
    @staticmethod
    def create_settings_tab(settings: Any) -> QWidget:
        """Create settings tab
        
        Args:
            settings: Application settings object
        
        Returns:
            Settings tab widget
        """
        try:
            return SettingsTabWidget(settings)
        except Exception as e:
            logger.error(f"Error creating settings tab: {str(e)}")
            raise
    
    @staticmethod
    def create_status_bar() -> Tuple[QStatusBar, QLabel]:
        """Create status bar"""
        try:
            status_bar = QStatusBar()
            status_label = QLabel(tr.get_text("status_ready", "Ready"))
            status_bar.addWidget(status_label)
            return status_bar, status_label
        except Exception as e:
            logger.error(f"Error creating status bar: {str(e)}")
            raise
    
    @staticmethod
    def create_logo_widget(logo_path: str = "gui/assets/logo.svg") -> QSvgWidget:
        """Create logo widget
        
        Args:
            logo_path: Path to the SVG logo file
            
        Returns:
            SVG widget with the logo
            
        Raises:
            FileNotFoundError: If the logo file does not exist
        """
        try:
            if not os.path.exists(logo_path):
                logger.warning(f"Logo file not found at {logo_path}, using default logo")
                # Create a default logo widget with text
                default_logo = QWidget()
                default_logo.setFixedSize(200, 50)
                layout = QVBoxLayout(default_logo)
                label = QLabel(tr.get_text("app_title", "Shop RPA"))
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)
                return default_logo
                
            logo = QSvgWidget(logo_path)
            logo.setFixedSize(200, 50)
            return logo
        except Exception as e:
            logger.error(f"Error creating logo widget: {str(e)}")
            raise
    
    @staticmethod
    def create_file_manager(settings: Any) -> FileManager:
        """Create file manager
        
        Args:
            settings: Application settings object
            
        Returns:
            File manager instance
        """
        try:
            return FileManager(settings)
        except Exception as e:
            logger.error(f"Error creating file manager: {str(e)}")
            raise
    
    @staticmethod
    def create_help_system(main_window: QWidget) -> HelpSystem:
        """Create help system
        
        Args:
            main_window: Main application window
            
        Returns:
            Help system instance
        """
        try:
            return HelpSystem(main_window)
        except Exception as e:
            logger.error(f"Error creating help system: {str(e)}")
            raise
    
    @staticmethod
    def create_performance_monitor() -> PerformanceMonitorWidget:
        """Create performance monitor"""
        try:
            return PerformanceMonitorWidget()
        except Exception as e:
            logger.error(f"Error creating performance monitor: {str(e)}")
            raise
    
    @staticmethod
    def create_toast_notification(parent: Optional[QWidget] = None) -> ToastNotification:
        """Create toast notification widget
        
        Args:
            parent: Parent widget
            
        Returns:
            Toast notification widget
            
        Example:
            # Create a toast notification
            toast = WidgetFactory.create_toast_notification(self)
            toast.show_message("Operation completed successfully", "success")
        """
        try:
            return ToastNotification(parent)
        except Exception as e:
            logger.error(f"Error creating toast notification: {str(e)}")
            raise
