"""Main window implementation for the shopping RPA system"""

import logging
import os
import subprocess
import time
import traceback
from typing import List, Optional
from pathlib import Path
import configparser

import psutil
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QDialog,
)
from PyQt5 import sip

from .i18n import translator as tr
from .log_handler import GUILogHandler
from .processing_thread import ProcessingThread
from .settings import Settings
from .styles import Colors, Styles, StyleTransition
from .widgets import WidgetFactory


class MainWindow(QMainWindow):
    """Main window of the application"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.program_root = Path(__file__).parent.parent.absolute()

        # Initialize settings
        self.settings = Settings()

        # Set debug mode based on config
        if self.config["GUI"]["DEBUG_MODE"]:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # 필수 속성 초기화
        self.input_file = None
        self.processing_thread = None
        self.auto_save_timer = None
        self.tab_widget = None
        self.log_area = None
        self.progress_bar = None
        self.status_label = None
        self.file_path_label = None
        self.drop_area = None
        self.start_button = None
        self.stop_button = None
        self.threads_spinbox = None
        self.memory_label = QLabel()

        # recent_files_list는 create_dashboard_tab에서 참조되므로 먼저 생성
        self.recent_files_list = QPlainTextEdit()
        self.recent_files_list.setReadOnly(True)

        # Initialize processor with config
        try:
            from core.processing.main_processor import ProductProcessor

            self.processor = ProductProcessor(self.config)

        except Exception as e:
            self.logger.error(f"프로세서 초기화 중 오류 발생: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "초기화 오류", f"프로세서 초기화 중 오류가 발생했습니다: {str(e)}"
            )

        # Set window properties
        self.setWindowTitle(tr.get_text("window_title", "상품 가격 비교 시스템"))
        self.setMinimumSize(800, 600)  # Keep minimum size for compatibility
        self.resize(
            int(self.config["GUI"]["WINDOW_WIDTH"]),
            int(self.config["GUI"]["WINDOW_HEIGHT"])
        )

        # Set application icon
        try:
            # Try PNG icon first (better Windows compatibility)
            icon_path = os.path.join(os.path.dirname(__file__), "assets", "app_icon.png")
            if os.path.exists(icon_path):
                app_icon = QIcon(icon_path)
                self.setWindowIcon(app_icon)
                # Ensure the icon is also set for the application
                from PyQt5.QtWidgets import QApplication
                QApplication.instance().setWindowIcon(app_icon)
            else:
                # Try SVG icon as fallback
                icon_path = os.path.join(os.path.dirname(__file__), "assets", "app_icon.svg")
                if os.path.exists(icon_path):
                    app_icon = QIcon(icon_path)
                    self.setWindowIcon(app_icon)
                    # Ensure the icon is also set for the application
                    from PyQt5.QtWidgets import QApplication
                    QApplication.instance().setWindowIcon(app_icon)
                else:
                    # Use default icon if no custom icon found
                    self.setWindowIcon(self.style().standardIcon(self.style().SP_ComputerIcon))
        except Exception as e:
            self.logger.warning(f"Failed to set application icon: {str(e)}")
            # Use default icon in case of error
            self.setWindowIcon(self.style().standardIcon(self.style().SP_ComputerIcon))

        # Enable drag & drop
        self.setAcceptDrops(True)

        # Initialize UI
        self.init_ui()

        # Apply theme based on settings
        self.apply_theme()

        # Setup auto-save
        self.setup_auto_save()

        # Setup memory monitoring
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(tr.get_text("Shop_RPA"))
        # Use config values for window size
        self.setMinimumSize(800, 600)  # Keep minimum size for compatibility
        self.resize(
            int(self.config["GUI"]["WINDOW_WIDTH"]),
            int(self.config["GUI"]["WINDOW_HEIGHT"])
        )

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_dashboard_tab()  # Add dashboard tab
        self.create_analysis_tab()
        self.create_settings_tab()
        self.create_help_tab()  # Add help tab

        # 상태바 설정
        status_bar = self.statusBar()
        status_bar.addPermanentWidget(self.memory_label)

        # Apply theme
        self.apply_theme()

        # Setup auto-save
        self.setup_auto_save()

        # Setup GUI logging
        self._setup_gui_logging()

        # Connect signals
        self.connect_signals()

        # Initialize settings
        self.initialize_settings()

        # Show the window
        self.show()

    def create_analysis_tab(self):
        """Create analysis tab for file processing"""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(15)

        # Create splitter for resizable sections
        self.splitter = QSplitter(Qt.Vertical)
        tab_layout.addWidget(self.splitter)

        # Top section (File handling and controls)
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create and add widgets using WidgetFactory
        self.file_group, self.drop_area, self.file_path_label = (
            WidgetFactory.create_file_group()
        )
        self.drop_area.clicked.connect(self.on_drop_area_clicked)
        top_layout.addWidget(self.file_group)
        
        # Add batch processing controls
        self.batch_files_list = QPlainTextEdit()
        self.batch_files_list.setReadOnly(True)
        self.batch_files_list.setPlaceholderText(tr.get_text("batch_files_placeholder"))
        self.batch_files_list.setMaximumHeight(100)
        
        batch_buttons_layout = QHBoxLayout()
        self.add_file_button = QPushButton(tr.get_text("add_file"))
        self.add_file_button.clicked.connect(self.on_add_file_clicked)
        self.clear_files_button = QPushButton(tr.get_text("clear_files"))
        self.clear_files_button.clicked.connect(self.on_clear_files_clicked)
        batch_buttons_layout.addWidget(self.add_file_button)
        batch_buttons_layout.addWidget(self.clear_files_button)
        
        batch_group = QGroupBox(tr.get_text("batch_processing"))
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.addWidget(self.batch_files_list)
        batch_layout.addLayout(batch_buttons_layout)
        
        top_layout.addWidget(batch_group)

        # --- Refactor Controls Group Layout ---
        self.controls_group = QGroupBox(tr.get_text("processing_controls"))
        controls_layout = QGridLayout(self.controls_group) # Use QGridLayout
        controls_layout.setSpacing(10) # Adjust spacing

        # Row 0: Threads
        threads_label = QLabel(tr.get_text("thread_count"))
        self.threads_spinbox = QSpinBox()
        self.threads_spinbox.setRange(1, os.cpu_count() or 4)
        self.threads_spinbox.setValue(self.settings.get("thread_count", 4))
        controls_layout.addWidget(threads_label, 0, 0)
        controls_layout.addWidget(self.threads_spinbox, 0, 1)

        # Row 1: Product Limit
        product_limit_label = QLabel(tr.get_text("max_products"))
        self.product_limit_spinbox = QSpinBox()
        self.product_limit_spinbox.setRange(0, 10000)  # 0 means no limit
        self.product_limit_spinbox.setToolTip(tr.get_text("max_products_tooltip"))
        self.product_limit_spinbox.setValue(0) # Default to no limit
        controls_layout.addWidget(product_limit_label, 1, 0)
        controls_layout.addWidget(self.product_limit_spinbox, 1, 1)

        # Row 2: Start/Stop Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton(tr.get_text("start"))
        self.stop_button = QPushButton(tr.get_text("stop"))
        self.stop_button.setEnabled(False) # Initially disabled
        Styles.apply_start_button_style(self.start_button)
        Styles.apply_stop_button_style(self.stop_button)
        button_layout.addStretch() # Push buttons to the right
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        controls_layout.addLayout(button_layout, 2, 0, 1, 3) # Span across columns

        # Set column stretch factors for alignment (make spinbox column expand)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 0) # Add a dummy column stretch if needed for alignment

        top_layout.addWidget(self.controls_group)
        # --- End Refactor ---

        # Add progress bar only if enabled in config
        if self.config["GUI"]["SHOW_PROGRESS_BAR"]:
            self.progress_group, self.progress_bar, self.status_label, self.step_indicator, self.resource_graph = (
                WidgetFactory.create_progress_group()
            )
            top_layout.addWidget(self.progress_group)
        else:
            self.progress_group = None
            self.progress_bar = None
            self.status_label = QLabel()
            self.step_indicator = None
            self.resource_graph = None
            top_layout.addWidget(self.status_label)

        # Add top section to splitter
        self.splitter.addWidget(top_section)

        # Bottom section (Log area)
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Create log area
        self.log_area = WidgetFactory.create_log_area()
        bottom_layout.addWidget(self.log_area)

        # Add bottom section to splitter
        self.splitter.addWidget(bottom_section)

        # Set initial splitter sizes (60% top, 40% bottom)
        self.splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])

        # Add tab to tab widget
        self.tab_widget.addTab(tab, tr.get_text("analysis"))

        return tab

    def create_settings_tab(self):
        """Create settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)

        # Add settings icon
        settings_icon = QSvgWidget(
            os.path.join(os.path.dirname(__file__), "assets", "settings.svg")
        )
        settings_icon.setFixedSize(48, 48)
        icon_layout = QHBoxLayout()
        icon_layout.addWidget(settings_icon, 0, Qt.AlignCenter)
        layout.addLayout(icon_layout)

        # General settings group
        general_group = QGroupBox(tr.get_text("general_settings"))
        general_layout = QVBoxLayout(general_group)
        general_layout.setSpacing(15)

        # Dark mode toggle
        dark_mode_layout = QHBoxLayout()
        dark_mode_label = QLabel(tr.get_text("dark_mode"))
        dark_mode_label.setFixedWidth(150)
        self.dark_mode_check = QCheckBox()
        Styles.apply_checkbox_style(self.dark_mode_check)
        dark_mode_layout.addWidget(dark_mode_label)
        dark_mode_layout.addWidget(self.dark_mode_check)
        dark_mode_layout.addStretch()
        general_layout.addLayout(dark_mode_layout)

        # Auto-save toggle
        auto_save_layout = QHBoxLayout()
        auto_save_label = QLabel(tr.get_text("auto_save"))
        auto_save_label.setFixedWidth(150)
        self.auto_save_check = QCheckBox()
        Styles.apply_checkbox_style(self.auto_save_check)
        auto_save_layout.addWidget(auto_save_label)
        auto_save_layout.addWidget(self.auto_save_check)
        auto_save_layout.addStretch()
        general_layout.addLayout(auto_save_layout)

        # Language selection
        language_layout = QHBoxLayout()
        language_label = QLabel(tr.get_text("language"))
        language_label.setFixedWidth(150)
        self.language_combo = QComboBox()
        self.language_combo.addItem("한국어", "ko_KR")
        self.language_combo.addItem("English", "en_US")
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        general_layout.addLayout(language_layout)

        layout.addWidget(general_group)

        # Processing settings group
        processing_group = QGroupBox(tr.get_text("processing_settings"))
        processing_layout = QVBoxLayout(processing_group)
        processing_layout.setSpacing(15)

        # Thread count setting
        thread_layout = QHBoxLayout()
        thread_label = QLabel(tr.get_text("default_thread_count"))
        thread_label.setFixedWidth(150)
        self.thread_spinbox = QSpinBox()
        self.thread_spinbox.setRange(1, os.cpu_count() or 4)
        thread_layout.addWidget(thread_label)
        thread_layout.addWidget(self.thread_spinbox)
        thread_layout.addStretch()
        processing_layout.addLayout(thread_layout)

        # Timeout setting
        timeout_layout = QHBoxLayout()
        timeout_label = QLabel(tr.get_text("request_timeout"))
        timeout_label.setFixedWidth(150)
        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setRange(10, 60)
        self.timeout_spinbox.setSuffix("s")
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.timeout_spinbox)
        timeout_layout.addStretch()
        processing_layout.addLayout(timeout_layout)

        # Similarity threshold setting
        similarity_layout = QHBoxLayout()
        similarity_label = QLabel(tr.get_text("similarity_threshold"))
        similarity_label.setFixedWidth(150)
        self.similarity_spinbox = QDoubleSpinBox()
        self.similarity_spinbox.setRange(0.0, 1.0)
        self.similarity_spinbox.setSingleStep(0.05)
        self.similarity_spinbox.setDecimals(2)
        self.similarity_spinbox.setValue(self.settings.get("similarity_threshold", 0.75))
        self.similarity_spinbox.setToolTip(tr.get_text("similarity_threshold_tooltip"))
        similarity_layout.addWidget(similarity_label)
        similarity_layout.addWidget(self.similarity_spinbox)
        similarity_layout.addStretch()
        processing_layout.addLayout(similarity_layout)

        layout.addWidget(processing_group)

        # Add apply button
        button_layout = QHBoxLayout()
        apply_button = QPushButton(tr.get_text("apply_settings"))
        apply_button.clicked.connect(self.apply_settings)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        layout.addLayout(button_layout)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Add tab to tab widget
        self.tab_widget.addTab(tab, tr.get_text("settings"))

        return tab

    def create_dashboard_tab(self):
        """Create the dashboard tab"""
        dashboard_tab = QWidget()
        layout = QVBoxLayout(dashboard_tab)

        # Recent files section
        recent_files_group = QGroupBox(tr.get_text("recent_files"))
        recent_files_layout = QVBoxLayout(recent_files_group)
        recent_files_layout.addWidget(self.recent_files_list)
        layout.addWidget(recent_files_group)

        # Statistics section (hidden)
        stats_layout = QHBoxLayout()

        # Products analyzed
        products_group = QGroupBox(tr.get_text("products_analyzed"))
        products_layout = QVBoxLayout(products_group)
        products_label = QLabel("0")
        products_label.setAlignment(Qt.AlignCenter)
        products_layout.addWidget(products_label)
        stats_layout.addWidget(products_group)
        products_group.hide()  # Hide instead of removing

        # Average price difference
        price_diff_group = QGroupBox(tr.get_text("avg_price_diff"))
        price_diff_layout = QVBoxLayout(price_diff_group)
        price_diff_label = QLabel("0%")
        price_diff_label.setAlignment(Qt.AlignCenter)
        price_diff_layout.addWidget(price_diff_label)
        stats_layout.addWidget(price_diff_group)
        price_diff_group.hide()  # Hide instead of removing

        # Processing time
        time_group = QGroupBox(tr.get_text("processing_time"))
        time_layout = QVBoxLayout(time_group)
        time_label = QLabel("0s")
        time_label.setAlignment(Qt.AlignCenter)
        time_layout.addWidget(time_label)
        stats_layout.addWidget(time_group)
        time_group.hide()  # Hide instead of removing

        layout.addLayout(stats_layout)

        # Add tab
        self.tab_widget.addTab(dashboard_tab, tr.get_text("dashboard"))

        return dashboard_tab

    def create_help_tab(self):
        """Create a help tab with documentation and tooltips"""
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        help_layout.setSpacing(20)
        
        # Store the help tab as an instance variable for access
        self.help_tab = help_tab
        
        # Add help icon
        help_icon = QSvgWidget(
            os.path.join(os.path.dirname(__file__), "assets", "help.svg")
        )
        help_icon.setFixedSize(48, 48)
        icon_layout = QHBoxLayout()
        icon_layout.addWidget(help_icon, 0, Qt.AlignCenter)
        help_layout.addLayout(icon_layout)
        
        # Create scrollable area for help content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(25)
        
        # Basic Usage section
        usage_group = QGroupBox(tr.get_text("basic_usage"))
        usage_layout = QVBoxLayout(usage_group)
        
        usage_text = QLabel(tr.get_text("basic_usage_text"))
        usage_text.setWordWrap(True)
        usage_text.setTextFormat(Qt.RichText)
        usage_layout.addWidget(usage_text)
        
        scroll_layout.addWidget(usage_group)
        
        # File Processing section
        proc_group = QGroupBox(tr.get_text("file_processing"))
        proc_layout = QVBoxLayout(proc_group)
        
        proc_text = QLabel(tr.get_text("file_processing_text"))
        proc_text.setWordWrap(True)
        proc_text.setTextFormat(Qt.RichText)
        proc_layout.addWidget(proc_text)
        
        scroll_layout.addWidget(proc_group)
        
        # Settings section
        settings_group = QGroupBox(tr.get_text("settings_help"))
        settings_layout = QVBoxLayout(settings_group)
        
        settings_text = QLabel(tr.get_text("settings_help_text"))
        settings_text.setWordWrap(True)
        settings_text.setTextFormat(Qt.RichText)
        settings_layout.addWidget(settings_text)
        
        scroll_layout.addWidget(settings_group)
        
        # Troubleshooting section
        troubleshoot_group = QGroupBox(tr.get_text("troubleshooting"))
        troubleshoot_layout = QVBoxLayout(troubleshoot_group)
        
        troubleshoot_text = QLabel(tr.get_text("troubleshooting_text"))
        troubleshoot_text.setWordWrap(True)
        troubleshoot_text.setTextFormat(Qt.RichText)
        troubleshoot_layout.addWidget(troubleshoot_text)
        
        scroll_layout.addWidget(troubleshoot_group)
        
        # Finish setting up scroll area
        scroll_area.setWidget(scroll_content)
        help_layout.addWidget(scroll_area)
        
        # Add tab to tab widget
        self.tab_widget.addTab(help_tab, tr.get_text("help"))
        
        return help_tab

    def apply_theme(self):
        """Apply current theme based on settings"""
        is_dark = self.config.get("GUI", {}).get("ENABLE_DARK_MODE", False)
        if isinstance(is_dark, str):
            is_dark = is_dark.lower() == "true"
            
        # Apply the right theme
        if is_dark:
            Styles.apply_dark_mode(self)
            # Update all child widgets
            for widget in self.findChildren(QWidget):
                if isinstance(widget, QGroupBox):
                    Styles.apply_group_box_style(widget)
                # Clean any custom stylesheets on other widgets
                elif widget.styleSheet():
                    clean_style = StyleTransition.remove_transition_property(widget.styleSheet())
                    widget.setStyleSheet(clean_style)
        else:
            Styles.apply_light_mode(self)
            # Update all child widgets
            for widget in self.findChildren(QWidget):
                if isinstance(widget, QGroupBox):
                    Styles.apply_group_box_style(widget)
                # Clean any custom stylesheets on other widgets
                elif widget.styleSheet():
                    clean_style = StyleTransition.remove_transition_property(widget.styleSheet())
                    widget.setStyleSheet(clean_style)

        # Update drop area style
        if hasattr(self, "drop_area"):
            self.drop_area.set_dark_mode(is_dark)

    def setup_auto_save(self):
        """Setup auto-save functionality"""
        interval = int(self.config["GUI"]["AUTO_SAVE_INTERVAL"])
        if interval > 0:
            self.auto_save_timer = QTimer()
            self.auto_save_timer.timeout.connect(self.auto_save)
            self.auto_save_timer.start(interval * 1000)

    def _setup_gui_logging(self):
        """Setup logging handler for GUI"""
        self.gui_handler = GUILogHandler()  # Initialize GUILogHandler correctly
        self.gui_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        # Connect the handler's signal to the main window's slot
        self.gui_handler.log_signal.connect(self.append_log_message)

        root_logger = logging.getLogger()
        root_logger.addHandler(self.gui_handler)
        root_logger.setLevel(self.settings.get_log_level())

        # Set max log lines from config
        self.max_log_lines = int(self.config["GUI"]["MAX_LOG_LINES"])

    def connect_signals(self):
        """Connect all signals to their slots"""
        self.start_button.clicked.connect(self.on_start_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.dark_mode_check.stateChanged.connect(self.toggle_dark_mode)
        self.auto_save_check.stateChanged.connect(self.toggle_auto_save)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def initialize_settings(self):
        """Initialize UI elements with current settings"""
        # Dark mode setting
        self.dark_mode_check.setChecked(self.settings.get("dark_mode", False))

        # Auto-save setting
        self.auto_save_check.setChecked(self.settings.get("auto_save", True))

        # Thread count settings
        default_threads = self.settings.get("thread_count", 4)
        self.threads_spinbox.setValue(default_threads)
        self.thread_spinbox.setValue(default_threads)

        # Timeout setting
        if (
            "PROCESSING" in self.config
            and "REQUEST_TIMEOUT" in self.config["PROCESSING"]
        ):
            self.timeout_spinbox.setValue(
                int(self.config["PROCESSING"]["REQUEST_TIMEOUT"])
            )
        else:
            self.timeout_spinbox.setValue(30)  # Default timeout

        # Language setting
        current_lang = self.settings.get("language", "ko_KR")
        index = self.language_combo.findData(current_lang)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

    def on_tab_changed(self, index):
        """Handle tab changes"""
        # Update recent files list when dashboard is selected
        if index == 0:  # Dashboard tab
            self.update_dashboard()

    def update_dashboard(self):
        """Update dashboard tab with current data"""
        try:
            # Update recent files list
            self.recent_files_list.clear()

            # Add recent files from settings
            recent_files = self.settings.get("recent_files", [])
            if recent_files:
                for file in recent_files:
                    if os.path.exists(file):
                        self.recent_files_list.appendPlainText(file)
            else:
                self.recent_files_list.appendPlainText(tr.get_text("no_recent_files"))

            # # Update statistics cards - Commented out historical metrics
            # # Find the statistics cards by traversing the tab's children
            # dashboard_tab = self.tab_widget.widget(0)

            # # Products count
            # products_count = self.settings.get("processed_count", 0)
            # for child in dashboard_tab.findChildren(QGroupBox):
            #     if child.title() == tr.get_text("products_analyzed"):
            #         for label in child.findChildren(QLabel):
            #             label.setText(str(products_count))

            # # Avg price diff (placeholder)
            # for child in dashboard_tab.findChildren(QGroupBox):
            #     if child.title() == tr.get_text("avg_price_diff"):
            #         for label in child.findChildren(QLabel):
            #             label.setText("~15%")  # Placeholder value

            # # Processing time
            # if self.settings.get("processed_count", 0) > 0:
            #     total_time = self.settings.get("total_processing_time", 0)
            #     avg_time = total_time / self.settings.get("processed_count", 1)
            #     time_text = f"{avg_time:.1f}s"

            #     for child in dashboard_tab.findChildren(QGroupBox):
            #         if child.title() == tr.get_text("processing_time"):
            #             for label in child.findChildren(QLabel):
            #                 label.setText(time_text)

        except Exception as e:
            logging.error(f"Error updating dashboard: {e}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            url = event.mimeData().urls()[0]
            if url.isLocalFile() and url.toLocalFile().lower().endswith(
                (".xlsx", ".xls")
            ):
                event.acceptProposedAction()
                self.drop_area.highlight_active()

    def dragLeaveEvent(self, event):
        self.drop_area.highlight_inactive()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop event"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".xlsx", ".xls")):
                self._add_file_to_batch(file_path)

    def _handle_file_selected(self, filepath: str):
        self.input_file = filepath
        self.file_path_label.setText(
            tr.get_text("selected_file", filename=os.path.basename(filepath))
        )
        self.status_label.setText(tr.get_text("ready_to_start"))
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_area.clear()
        logging.info(tr.get_text("file_selected", filename=filepath))

        # Add to recent files list
        recent_files = self.settings.get("recent_files", [])

        # Remove file if it's already in the list
        if filepath in recent_files:
            recent_files.remove(filepath)

        # Add to the beginning of the list
        recent_files.insert(0, filepath)

        # Keep only the 5 most recent files
        recent_files = recent_files[:5]

        # Save to settings
        self.settings.set("recent_files", recent_files)
        self.settings.save_settings()

        # Update dashboard if it's visible
        if self.tab_widget.currentIndex() == 0:
            self.update_dashboard()

    @pyqtSlot()
    def on_start_clicked(self):
        if not self.input_file or not os.path.exists(self.input_file):
            QMessageBox.warning(
                self, tr.get_text("warning"), tr.get_text("no_valid_file")
            )
            return

        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.information(
                self, tr.get_text("info"), tr.get_text("already_processing")
            )
            return

        self._start_processing()

    def _start_processing(self):
        """Start processing selected files"""
        try:
            # Update state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText(tr.get_text("starting"))
            self.progress_bar.setValue(0)
            self.log_area.clear()
            logging.info(tr.get_text("analysis_started"))
            
            # Record start time for statistics
            self.processing_start_time = time.time()

            # Determine the files to process
            input_files = self.input_files if hasattr(self, 'input_files') and self.input_files else [self.input_file]
            
            # Get the product limit
            product_limit = self.product_limit_spinbox.value() if self.product_limit_spinbox.value() > 0 else None
            if product_limit:
                logging.info(f"상품 처리 수 제한 설정됨: {product_limit}개")

            # Create and start processing thread
            self.processing_thread = ProcessingThread(
                self.processor, 
                input_files, 
                product_limit=product_limit
            )
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.status_updated.connect(self.update_status)
            self.processing_thread.log_message.connect(self.append_log)
            self.processing_thread.processing_complete.connect(self.on_processing_complete)
            self.processing_thread.processing_error.connect(self.on_processing_error)
            self.processing_thread.finished.connect(self.on_thread_finished)

            self.processing_thread.start()

        except Exception as e:
            logging.error(f"Error starting processing: {e}")
            self.status_label.setText(tr.get_text("error_occurred"))
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    @pyqtSlot()
    def on_stop_clicked(self):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                tr.get_text("stop_processing"),
                tr.get_text("confirm_stop"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.stop_button.setEnabled(False)
                self.status_label.setText(tr.get_text("stopping"))
                logging.info(tr.get_text("user_requested_stop"))

    @pyqtSlot(int, int)
    def update_progress(self, current_item, total_items):
        """Update progress bar value and ensure it's visible"""
        if not self.progress_bar.isVisible():
            self.progress_bar.show()

        if total_items > 0:
            percentage = int((current_item / total_items) * 100)
            # Ensure percentage is within 0-100 bounds
            percentage = max(0, min(percentage, 100))
            
            # Smoothly animate progress bar value if change is significant
            current_value = self.progress_bar.value()
            if abs(percentage - current_value) > 3:  # Only animate if difference is significant
                # Use property animation for smooth transition
                self.progress_bar.setValue(percentage)
            else:
                self.progress_bar.setValue(percentage)
            
            # Update progress bar text to show both percentage and count
            self.progress_bar.setFormat(f"{percentage}% ({current_item}/{total_items})")

            # During processing, update drop area to show loading
            if percentage > 0 and percentage < 100 and hasattr(self, 'drop_area'):
                if not hasattr(self, 'loading_icon') or self.loading_icon is None:
                    self.loading_icon = QSvgWidget(os.path.join(os.path.dirname(__file__), "assets", "loading.svg"))
                    self.loading_icon.setFixedSize(48, 48)
                    # Replace the file icon with the loading icon temporarily
                    if hasattr(self.drop_area, 'file_icon') and self.drop_area.file_icon:
                        self.drop_area.file_icon.hide()
                        # Find the layout containing the file icon
                        icon_layout = None
                        for i in range(self.drop_area.layout().count()):
                            item = self.drop_area.layout().itemAt(i)
                            if isinstance(item, QHBoxLayout):
                                icon_layout = item
                                break
                        
                        if icon_layout:
                            # Add loading icon to the layout
                            icon_layout.insertWidget(1, self.loading_icon)
                            self.loading_icon.show()
                            
                # Update the hint text to show progress
                if hasattr(self.drop_area, 'hint_label'):
                    self.drop_area.hint_label.setText(f"{tr.get_text('processing')}: {percentage}%")

            # Update status label with count
            status_text = tr.get_text("processing_progress", current=current_item, total=total_items)
            self.status_label.setText(status_text)
            
            # Update step indicator based on progress
            if hasattr(self, 'step_indicator'):
                if percentage <= 10:
                    self.step_indicator.set_current_step(0)  # Load step
                    self.step_indicator.set_completed_steps(0)
                elif percentage <= 90:
                    self.step_indicator.set_current_step(1)  # Process step
                    self.step_indicator.set_completed_steps(1)
                else:
                    self.step_indicator.set_current_step(2)  # Save step
                    self.step_indicator.set_completed_steps(2)
            
            # Update resource graph with real-time data
            if hasattr(self, 'resource_graph'):
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent(interval=0.1)
                    self.resource_graph.update_data(memory_mb, cpu_percent)
                except ImportError:
                    pass  # psutil not available
                except Exception as e:
                    logging.debug(f"Resource graph update error: {str(e)}")
        else:
            # Handle case where total_items is 0 or less (e.g., empty file)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0% (0/0)")
            self.status_label.setText(tr.get_text("processing_starting")) # Or a more specific message

        self.progress_bar.repaint()  # Force immediate update

    @pyqtSlot(str)
    def update_status(self, message):
        self.status_label.setText(message)

    @pyqtSlot(str)
    def append_log(self, message):
        self.log_area.appendPlainText(message)

    @pyqtSlot(str, str)
    def on_processing_complete(self, primary_path, secondary_path):
        """Handle processing completion"""
        try:
            self._show_completion_message(primary_path, secondary_path)
            self._show_results_summary([primary_path, secondary_path])
            
            # Show success toast notification
            toast = WidgetFactory.create_toast_notification(self)
            toast.show_message(
                tr.get_text("processing_complete", "Processing completed successfully"),
                "success"
            )
        except Exception as e:
            self.logger.error(f"Error handling processing completion: {str(e)}")
            self._show_error_message(str(e))

    def on_processing_error(self, error_message):
        """Handle processing error"""
        try:
            self.status_label.setText(tr.get_text("status_error", "Error"))
            self.append_log(f"Error: {error_message}")
            
            # Show error toast notification
            toast = WidgetFactory.create_toast_notification(self)
            toast.show_message(
                tr.get_text("processing_error", f"Error: {error_message}"),
                "error"
            )
        except Exception as e:
            self.logger.error(f"Error handling processing error: {str(e)}")
            self._show_error_message(str(e))

    def _show_error_message(self, error_message):
        """Show error message in a user-friendly way"""
        try:
            # Show error toast notification
            toast = WidgetFactory.create_toast_notification(self)
            toast.show_message(
                tr.get_text("error_occurred", f"An error occurred: {error_message}"),
                "error"
            )
            
            # Log the error
            self.logger.error(error_message)
            self.append_log(error_message)
        except Exception as e:
            self.logger.error(f"Error showing error message: {str(e)}")
            # Fallback to basic error display
            self.status_label.setText(f"Error: {str(e)}")

    def _show_completion_message(self, primary_path, secondary_path):
        """Show completion message with toast notification"""
        try:
            message = tr.get_text(
                "processing_complete_details",
                "Processing completed:\nPrimary: {primary}\nSecondary: {secondary}"
            ).format(primary=primary_path, secondary=secondary_path)
            
            # Show success toast notification
            toast = WidgetFactory.create_toast_notification(self)
            toast.show_message(message, "success")
            
            self.status_label.setText(tr.get_text("status_completed", "Completed"))
            self.append_log(message)
        except Exception as e:
            self.logger.error(f"Error showing completion message: {str(e)}")
            self._show_error_message(str(e))

    def open_results_folder(self):
        """Open the results folder in file explorer"""
        try:
            output_dir = self.program_root / "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            os.startfile(str(output_dir))
            logging.info(tr.get_text("opened_output_dir", path=str(output_dir)))
        except Exception as e:
            logging.warning(tr.get_text("failed_to_open_dir", error=str(e)))

    @pyqtSlot()
    def on_thread_finished(self):
        """Handle thread completion"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.processing_thread = None
        
        # Restore the file icon if we were showing a loading animation
        if hasattr(self, 'loading_icon') and self.loading_icon:
            self.loading_icon.hide()
            self.loading_icon.deleteLater()
            self.loading_icon = None
            
            # Show the original file icon again
            if hasattr(self.drop_area, 'file_icon'):
                self.drop_area.file_icon.show()
                
            # Reset the hint text
            if hasattr(self.drop_area, 'hint_label'):
                self.drop_area.hint_label.setText(tr.get_text("file_drag_hint"))

    def update_memory_usage(self):
        """Update memory usage in status bar"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_text = tr.get_text("memory_usage", memory=memory_mb)

            # 메모리 라벨 업데이트
            self.memory_label.setText(memory_text)

            # 상태바 메시지도 업데이트 (기존 호환성 유지)
            self.statusBar().showMessage(memory_text)
        except Exception as e:
            self.logger.error(f"메모리 사용량 업데이트 중 오류 발생: {str(e)}")
            self.memory_label.setText("Memory: Unknown")

    def toggle_dark_mode(self, enabled):
        """Toggle dark mode and save setting"""
        if 'GUI' not in self.config:
            self.config['GUI'] = {}
        self.config['GUI']['ENABLE_DARK_MODE'] = str(enabled).lower()
        self.settings.set("dark_mode", enabled)
        
        # Apply theme update with CSS transition cleanup
        self.apply_theme()
        
        # Save config to file using configparser
        config_parser = configparser.ConfigParser()
        # 먼저 기존 config 파일을 읽습니다
        config_parser.read("config.ini", encoding="utf-8")
        # 새로운 설정을 업데이트합니다
        if 'GUI' not in config_parser:
            config_parser['GUI'] = {}
        config_parser['GUI']['ENABLE_DARK_MODE'] = str(enabled).lower()
        # 변경된 설정을 파일에 저장합니다
        with open("config.ini", "w", encoding="utf-8") as f:
            config_parser.write(f)

    def toggle_auto_save(self, enabled):
        self.settings.set("auto_save", enabled)

    def auto_save(self):
        """Auto-save current state"""
        if (
            self.input_file
            and self.processing_thread
            and self.processing_thread.isRunning()
        ):
            try:
                self.save_progress()
            except Exception as e:
                logging.error(tr.get_text("auto_save_failed", error=str(e)))

    def save_progress(self):
        """Save current processing progress"""
        # Implement progress saving logic here
        pass

    def closeEvent(self, event):
        """Handle window close event"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                tr.get_text("confirm_exit"),
                tr.get_text("exit_while_processing"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                logging.info(tr.get_text("user_requested_exit"))
                self.processing_thread.stop()
                self.processing_thread.wait(2000)  # Wait max 2 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def apply_settings(self):
        """Apply current settings"""
        try:
            # Save dark mode setting
            self.settings.set("dark_mode", self.dark_mode_check.isChecked())
            self.apply_theme()

            # Save auto-save setting
            self.settings.set("auto_save", self.auto_save_check.isChecked())
            self.setup_auto_save()

            # Save language setting
            old_language = self.settings.get("language")
            new_language = self.language_combo.currentData()
            language_changed = old_language != new_language
            self.settings.set("language", new_language)

            if language_changed:
                QMessageBox.information(
                    self,
                    tr.get_text("language_changed"),
                    tr.get_text("restart_required"),
                )

            # Save thread count and sync with analysis tab
            self.settings.set("thread_count", self.thread_spinbox.value())
            self.threads_spinbox.setValue(self.thread_spinbox.value())
            
            # Save timeout and update config
            self.settings.set("timeout", self.timeout_spinbox.value())
            if "PROCESSING" in self.config:
                self.config["PROCESSING"]["REQUEST_TIMEOUT"] = str(
                    self.timeout_spinbox.value()
                )
            
            # Save similarity threshold
            self.settings.set("similarity_threshold", self.similarity_spinbox.value())

            # Save settings
            self.settings.save_settings()

            # Show success message
            QMessageBox.information(
                self, tr.get_text("settings_saved"), tr.get_text("settings_applied")
            )

        except Exception as e:
            self.logger.error(f"설정 저장 실패: {str(e)}", exc_info=True)
            QMessageBox.warning(
                self, tr.get_text("warning"), f"{tr.get_text('settings_save_failed')}: {str(e)}"
            )

    def on_drop_area_clicked(self):
        """Handle click on drop area to open file dialog"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            tr.get_text("select_file"),
            "",
            tr.get_text("excel_files") + " (*.xlsx *.xls)",
        )
        if filepath:
            self._handle_file_selected(filepath)

    @pyqtSlot(list)
    def processing_finished(self, output_files):
        """Handle processing finished for one or more files"""
        self.update_status(tr.get_text("processing_finished"))
        self.progress_bar.setValue(100)  # Ensure progress bar shows completion
        
        # Calculate processing time
        if hasattr(self, 'processing_start_time'):
            elapsed_time = time.time() - self.processing_start_time
            self.logger.info(f"처리 시간: {elapsed_time:.2f}초")
        
        # Reset UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if not output_files:
            QMessageBox.warning(
                self,
                tr.get_text("warning"),
                tr.get_text("no_reports_generated")
            )
            return
            
        # Create and show results summary dialog
        self._show_results_summary(output_files)
        
        # Add to recent files
        for file_path in self.input_files if hasattr(self, 'input_files') else [self.input_file]:
            if file_path and os.path.exists(file_path):
                self._add_to_recent_files(file_path)
        
        # Update dashboard
        self.update_dashboard()
            
    def _show_results_summary(self, output_files):
        """Show a summary dialog of processing results with visualization options"""
        summary_dialog = QDialog(self)
        summary_dialog.setWindowTitle(tr.get_text("processing_complete"))
        summary_dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(summary_dialog)
        
        # Header with success message
        header = QLabel(tr.get_text("analysis_complete_detail"))
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        layout.addWidget(header)
        
        # File list with links
        file_group = QGroupBox(tr.get_text("generated_reports"))
        file_layout = QVBoxLayout(file_group)
        
        for output_file in output_files:
            file_name = os.path.basename(output_file)
            file_row = QHBoxLayout()
            
            # File icon
            icon_label = QLabel()
            icon_label.setPixmap(self.style().standardPixmap(self.style().SP_FileIcon))
            
            # File name with link
            file_link = QPushButton(file_name)
            file_link.setStyleSheet("text-align: left; border: none; text-decoration: underline; color: blue;")
            file_link.setCursor(Qt.PointingHandCursor)
            
            # Use lambda with default arg to avoid late binding issues
            file_link.clicked.connect(lambda checked=False, path=output_file: self._open_file(path))
            
            file_row.addWidget(icon_label)
            file_row.addWidget(file_link, 1)  # Stretch to fill space
            file_layout.addLayout(file_row)
        
        layout.addWidget(file_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        open_folder_btn = QPushButton(tr.get_text("open_results_folder"))
        open_folder_btn.clicked.connect(self.open_results_folder)
        
        close_btn = QPushButton(tr.get_text("close"))
        close_btn.clicked.connect(summary_dialog.accept)
        
        button_layout.addWidget(open_folder_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Show the dialog
        summary_dialog.exec_()
    
    def _open_file(self, file_path):
        """Open a file with the system's default application"""
        if not os.path.exists(file_path):
            QMessageBox.warning(
                self,
                tr.get_text("warning"),
                tr.get_text("file_not_found")
            )
            return
            
        try:
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS, Linux
                subprocess.call(('xdg-open', file_path))
            self.logger.info(f"File opened: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to open file: {e}")
            QMessageBox.critical(
                self,
                tr.get_text("error"),
                f"{tr.get_text('failed_to_open_file')}: {str(e)}"
            )
            
    def _add_to_recent_files(self, file_path):
        """Add a file to the recent files list"""
        recent_files = self.settings.get("recent_files", [])
        
        # Add to beginning if not already there, otherwise move to beginning
        if file_path in recent_files:
            recent_files.remove(file_path)
        recent_files.insert(0, file_path)
        
        # Limit to 10 recent files
        recent_files = recent_files[:10]
        
        # Save to settings
        self.settings.set("recent_files", recent_files)

    def append_log_message(self, msg):
        """Append a log message to the log area safely from any thread."""
        try:
            self.log_area.appendPlainText(msg)
            # Limit the number of lines in the log area based on config
            if self.log_area.document().lineCount() > self.max_log_lines:
                cursor = self.log_area.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.movePosition(
                    cursor.Down,
                    cursor.KeepAnchor,
                    self.log_area.document().lineCount() - self.max_log_lines,
                )
                cursor.removeSelectedText()
                cursor.movePosition(cursor.End)
                self.log_area.setTextCursor(cursor)

            self.log_area.ensureCursorVisible()  # Ensure the latest message is visible
        except Exception as e:
            # Fallback logging if GUI update fails
            print(f"Fallback log: {msg}\nError updating log area: {e}")

    def resizeEvent(self, event):
        """Handle window resize events to adjust layouts"""
        super().resizeEvent(event)
        
        # Adjust splitter sizes based on window height
        if hasattr(self, 'splitter'):
            window_height = self.height()
            self.splitter.setSizes([int(window_height * 0.6), int(window_height * 0.4)])

    def on_add_file_clicked(self):
        """Add file to batch processing queue"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr.get_text("select_file"), "", tr.get_text("excel_files")
        )
        if file_path:
            self._add_file_to_batch(file_path)
    
    def on_clear_files_clicked(self):
        """Clear all files from batch processing queue"""
        self.batch_files_list.clear()
        self.input_files = []
        self.status_label.setText(tr.get_text("batch_cleared"))
    
    def _add_file_to_batch(self, file_path):
        """Add a file to the batch processing queue"""
        if not hasattr(self, 'input_files'):
            self.input_files = []
            
        # Check if file is already in the list
        if file_path in self.input_files:
            return
            
        # Add to the list and update the display
        self.input_files.append(file_path)
        current_text = self.batch_files_list.toPlainText()
        if current_text:
            self.batch_files_list.setPlainText(current_text + "\n" + file_path)
        else:
            self.batch_files_list.setPlainText(file_path)
            
        # Enable start button if there are files
        self.start_button.setEnabled(True)
        self.status_label.setText(tr.get_text("files_in_batch", count=len(self.input_files)))


# Custom logging handler to emit logs to the GUI text area
class GUILogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)  # Define a signal that carries a string

    def __init__(self, parent=None):
        logging.Handler.__init__(self)
        QObject.__init__(self, parent)

    def emit(self, record):
        try:
            msg = self.format(record)
            # 객체가 아직 존재하는지 확인
            if not sip.isdeleted(self):
                self.log_signal.emit(msg)
        except Exception:
            self.handleError(record)
