"""Main window implementation for the shopping RPA system"""

import logging
import os
import subprocess
import time
import traceback
from typing import List, Optional

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
)
from PyQt5 import sip

from .i18n import translator as tr
from .log_handler import GUILogHandler
from .processing_thread import ProcessingThread
from .settings import Settings
from .styles import Colors, Styles
from .widgets import WidgetFactory


class MainWindow(QMainWindow):
    """Main window of the application"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize settings
        self.settings = Settings()

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
        self.setGeometry(
            100,
            100,
            int(config["GUI"]["WINDOW_WIDTH"]),
            int(config["GUI"]["WINDOW_HEIGHT"]),
        )

        # Set application icon
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "app_icon.svg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

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
        self.setMinimumSize(1200, 800)

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
        splitter = QSplitter(Qt.Vertical)
        tab_layout.addWidget(splitter)

        # Top section (File handling and controls)
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create and add widgets using WidgetFactory
        self.file_group, self.drop_area, self.file_path_label = (
            WidgetFactory.create_file_group()
        )
        self.drop_area.clicked.connect(self.on_drop_area_clicked)  # Add click handler
        top_layout.addWidget(self.file_group)

        (
            self.controls_group,
            self.start_button,
            self.stop_button,
            self.threads_spinbox,
        ) = WidgetFactory.create_controls_group()
        # Add product limit spinbox to controls group
        controls_layout = self.controls_group.layout()
        product_limit_layout = QHBoxLayout()
        product_limit_label = QLabel(tr.get_text("max_products"))
        product_limit_label.setFixedWidth(150)  # Align with other labels
        self.product_limit_spinbox = QSpinBox()
        self.product_limit_spinbox.setRange(0, 10000)  # 0 means no limit
        self.product_limit_spinbox.setToolTip(tr.get_text("max_products_tooltip"))
        self.product_limit_spinbox.setValue(0) # Default to no limit
        product_limit_layout.addWidget(product_limit_label)
        product_limit_layout.addWidget(self.product_limit_spinbox)
        product_limit_layout.addStretch()
        # Insert the new layout before the stretch in the original controls layout if it exists
        if isinstance(controls_layout, QVBoxLayout):
            # Find the stretch item and insert before it, or just add if no stretch
            stretch_index = -1
            for i in range(controls_layout.count()):
                item = controls_layout.itemAt(i)
                if isinstance(item, type(controls_layout.itemAt(0))) and item.spacerItem(): # Check if it's a spacer item
                   stretch_index = i
                   break
            if stretch_index != -1:
                controls_layout.insertLayout(stretch_index, product_limit_layout)
            else:
                controls_layout.addLayout(product_limit_layout) # Add if no stretch found
        else: # Fallback if layout is not QVBoxLayout as expected
             controls_layout.addLayout(product_limit_layout)

        top_layout.addWidget(self.controls_group)

        self.progress_group, self.progress_bar, self.status_label = (
            WidgetFactory.create_progress_group()
        )
        top_layout.addWidget(self.progress_group)

        # Add top section to splitter
        splitter.addWidget(top_section)

        # Bottom section (Log area)
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Create log area
        self.log_area = WidgetFactory.create_log_area()
        bottom_layout.addWidget(self.log_area)

        # Add bottom section to splitter
        splitter.addWidget(bottom_section)

        # Set initial splitter sizes (60% top, 40% bottom)
        splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])

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

    def apply_theme(self):
        """Apply current theme based on settings"""
        is_dark = self.settings.get("dark_mode")
        if is_dark:
            # Set BORDER attribute for dark mode
            Colors.BORDER = Colors.BORDER_DARK
            Styles.apply_dark_mode(self)
            # Apply dark mode to specific widgets
            self.tab_widget.setStyleSheet(
                f"""
                QTabWidget::pane {{
                    border: 1px solid {Colors.BORDER};
                    background: {Colors.BACKGROUND_DARK};
                }}
                QTabBar::tab {{
                    background: {Colors.SIDEBAR_DARK};
                    color: {Colors.TEXT_DARK};
                    padding: 8px 20px;
                    border: 1px solid {Colors.BORDER};
                    border-bottom: none;
                }}
                QTabBar::tab:selected {{
                    background: {Colors.PRIMARY};
                    color: white;
                }}
            """
            )
            self.log_area.setStyleSheet(
                f"""
                QPlainTextEdit {{
                    background-color: {Colors.SIDEBAR_DARK};
                    color: {Colors.TEXT_DARK};
                    border: 1px solid {Colors.BORDER};
                    border-radius: 5px;
                }}
            """
            )
            self.file_path_label.setStyleSheet(f"color: {Colors.TEXT_DARK};")
            self.status_label.setStyleSheet(f"color: {Colors.TEXT_DARK};")
            self.memory_label.setStyleSheet(f"color: {Colors.TEXT_DARK};")
        else:
            # Set BORDER attribute for light mode
            Colors.BORDER = Colors.BORDER_LIGHT
            Styles.apply_light_mode(self)
            # Apply light mode to specific widgets
            self.tab_widget.setStyleSheet("")
            self.log_area.setStyleSheet("")
            self.file_path_label.setStyleSheet("")
            self.status_label.setStyleSheet("")
            self.memory_label.setStyleSheet("")

        # Update drop area style
        if hasattr(self, "drop_area"):
            self.drop_area.set_dark_mode(is_dark)

    def setup_auto_save(self):
        """Setup auto-save functionality"""
        interval = self.settings.get("auto_save_interval", 300)
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
        self.drop_area.highlight_inactive()
        url = event.mimeData().urls()[0]
        filepath = url.toLocalFile()
        self._handle_file_selected(filepath)

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
        """Start the processing thread"""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(tr.get_text("starting"))
        self.progress_bar.setValue(0)
        self.log_area.clear()
        logging.info(tr.get_text("analysis_started"))

        # Record start time for statistics
        self.processing_start_time = time.time()

        # Get product limit from spinbox
        product_limit = self.product_limit_spinbox.value()
        if product_limit <= 0: # Treat 0 or less as no limit
            product_limit = None
        else:
             logging.info(f"상품 처리 수 제한 설정됨: {product_limit}개")

        self.processing_thread = ProcessingThread(
            self.processor, self.input_file, product_limit
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.log_message.connect(self.append_log)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.finished.connect(self.on_thread_finished)

        self.processing_thread.start()

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

    @pyqtSlot(int)
    def update_progress(self, value):
        """Update progress bar value and ensure it's visible"""
        if not self.progress_bar.isVisible():
            self.progress_bar.show()
        if value < 0:
            value = 0
        elif value > 100:
            value = 100
        self.progress_bar.setValue(value)
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
            self.status_label.setText(tr.get_text("processing_finished"))
            self._show_completion_message(primary_path, secondary_path)

            # # Commented out historical metrics update
            # # Increment processed files count
            # processed_count = self.settings.get("processed_count", 0) + 1
            # self.settings.set("processed_count", processed_count)

            # # Record processing time
            # if hasattr(self, 'processing_start_time') and self.processing_start_time:
            #     elapsed_time = time.time() - self.processing_start_time

            #     # Update average processing time
            #     total_time = self.settings.get("total_processing_time", 0) + elapsed_time
            #     self.settings.set("total_processing_time", total_time)

            #     # Store this processing time
            #     self.settings.set("last_processing_time", elapsed_time)

            # # Save settings
            # self.settings.save_settings()

            # # Update dashboard if it's visible
            # if self.tab_widget.currentIndex() == 0:
            #     self.update_dashboard()

        except Exception as e:
            logging.error(f"Error updating statistics: {e}")

    def _show_completion_message(self, primary_path, secondary_path):
        """Show completion message with report details"""
        message = tr.get_text("analysis_complete_detail") + "\n\n"

        if primary_path and os.path.exists(primary_path):
            message += (
                tr.get_text("primary_report", filename=os.path.basename(primary_path))
                + "\n"
            )

        if secondary_path and os.path.exists(secondary_path):
            message += (
                tr.get_text(
                    "secondary_report", filename=os.path.basename(secondary_path)
                )
                + "\n"
            )

        message += f"\n{tr.get_text('output_directory', 
                                   path=os.path.dirname(primary_path or secondary_path))}"

        QMessageBox.information(self, tr.get_text("complete"), message)

    def _open_output_directory(self, report_path):
        """Open the output directory in file explorer"""
        try:
            output_dir = os.path.dirname(report_path)
            os.startfile(output_dir)
            logging.info(tr.get_text("opened_output_dir", path=output_dir))
        except Exception as e:
            logging.warning(tr.get_text("failed_to_open_dir", error=str(e)))

    @pyqtSlot(str)
    def on_processing_error(self, error_message):
        self.status_label.setText(tr.get_text("error_occurred"))
        QMessageBox.critical(
            self,
            tr.get_text("error"),
            tr.get_text("processing_error_detail", error=error_message),
        )

    @pyqtSlot()
    def on_thread_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.processing_thread = None

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
        self.settings.set("dark_mode", enabled)
        self.apply_theme()

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

    def open_results_folder(self):
        """Open the results folder in file explorer"""
        try:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "output"
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            os.startfile(output_dir)
            logging.info(tr.get_text("opened_output_dir", path=output_dir))
        except Exception as e:
            logging.warning(tr.get_text("failed_to_open_dir", error=str(e)))

    def apply_settings(self):
        """Apply settings from the settings tab"""
        # Apply theme
        self.settings.set("dark_mode", self.dark_mode_check.isChecked())
        self.apply_theme()

        # Apply auto-save
        self.settings.set("auto_save", self.auto_save_check.isChecked())

        # Apply thread count
        self.settings.set("thread_count", self.thread_spinbox.value())
        self.threads_spinbox.setValue(self.thread_spinbox.value())

        # Apply timeout
        if "PROCESSING" in self.config:
            self.config["PROCESSING"]["REQUEST_TIMEOUT"] = str(
                self.timeout_spinbox.value()
            )

        # Apply language
        new_lang = self.language_combo.currentData()
        if new_lang != self.settings.get("language", "ko_KR"):
            self.settings.set("language", new_lang)
            QMessageBox.information(
                self, tr.get_text("language_changed"), tr.get_text("restart_required")
            )

        # Save settings
        self.settings.save_settings()

        QMessageBox.information(
            self, tr.get_text("settings_saved"), tr.get_text("settings_applied")
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

    def processing_finished(self, output_files):
        """처리 완료 시 호출되는 메서드"""
        self.progress_bar.hide()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

        if not output_files:
            self.status_label.setText("처리 실패: 출력 파일이 생성되지 않았습니다.")
            return

        elapsed_time = time.time() - self.processing_start_time
        self.status_label.setText(f"처리 완료! (소요 시간: {elapsed_time:.1f}초)")

        # 결과 표시
        result_message = f"{len(output_files)}개 파일이 생성되었습니다:\n\n"
        for file in output_files:
            result_message += f"- {file}\n"

        # 사용자에게 결과 파일 경로 알림
        QMessageBox.information(self, "처리 완료", result_message)

        # 결과 파일 열기
        for file in output_files:
            if file and os.path.exists(file):
                try:
                    # Windows에서는 start 명령 사용, 다른 OS는 적절한 명령 사용
                    if os.name == "nt":
                        os.startfile(file)
                    elif os.name == "posix":
                        subprocess.call(("xdg-open", file))
                    else:
                        self.logger.warning(
                            f"알 수 없는 OS: {os.name}, 파일을 열 수 없습니다."
                        )
                except Exception as e:
                    self.logger.error(f"결과 파일 열기 실패: {str(e)}", exc_info=True)
                    QMessageBox.warning(
                        self, "파일 열기 오류", f"결과 파일을 열 수 없습니다: {str(e)}"
                    )

    @pyqtSlot(str)  # Decorator to mark this as a slot that accepts a string
    def append_log_message(self, msg):
        """Append a log message to the log area safely from any thread."""
        try:
            self.log_area.appendPlainText(msg)
            # Optional: Limit the number of lines in the log area
            max_lines = self.settings.get("max_log_lines", 1000)
            if self.log_area.document().lineCount() > max_lines:
                cursor = self.log_area.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.movePosition(
                    cursor.Down,
                    cursor.KeepAnchor,
                    self.log_area.document().lineCount() - max_lines,
                )
                cursor.removeSelectedText()
                cursor.movePosition(cursor.End)
                self.log_area.setTextCursor(cursor)

            self.log_area.ensureCursorVisible()  # Ensure the latest message is visible
        except Exception as e:
            # Fallback logging if GUI update fails
            print(f"Fallback log: {msg}\nError updating log area: {e}")


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
