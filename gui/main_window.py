import sys
import os
import traceback
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QProgressBar, QMessageBox, 
                             QPlainTextEdit, QFileDialog, QSplitter, QFrame,
                             QStyle, QStyleFactory, QComboBox, QSpinBox,
                             QCheckBox, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSlot, QThread, pyqtSignal, QMetaObject, Q_ARG
from PyQt5.QtGui import QPalette, QColor, QFont, QIcon
from core.processing import Processor
import logging

# Worker thread for running the processing task
class ProcessingThread(QThread):
    # Signals
    progress_updated = pyqtSignal(int) # Current progress percentage
    status_updated = pyqtSignal(str) # Status message
    log_message = pyqtSignal(str)   # Log message
    processing_complete = pyqtSignal(str, str) # Primary and secondary report paths
    processing_error = pyqtSignal(str) # Error message
    memory_usage_updated = pyqtSignal(float)  # New signal for memory usage

    def __init__(self, processor: Processor, input_file: str):
        super().__init__()
        self.processor = processor
        self.input_file = input_file
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.status_updated.emit("파일 로딩 중...")
            # Read input file to get total count first
            # This might need adjustment based on how Processor counts items
            import pandas as pd
            try:
                df = pd.read_excel(self.input_file)
                total_items = len(df)
            except Exception as e:
                 self.log_message.emit(f"입력 파일 로딩 오류: {e}")
                 total_items = 0 # Cannot determine total
                 raise # Propagate error

            self.status_updated.emit(f"{total_items}개 상품 처리 시작...")
            self.progress_updated.emit(0)

            # --- Connect Processor logging to GUI --- 
            # This requires modifying Processor or using a custom handler
            # For now, we rely on Processor logging internally and signals
            # Example: We could pass a callback to Processor to update progress/log
            # processor.set_progress_callback(self.update_progress)
            # -----------------------------------------

            # Run the main processing task
            primary_report_path, secondary_report_path = self.processor.process_file(self.input_file)
            
            if self._is_running:  # Only emit if not stopped
                self.progress_updated.emit(100)
                self.status_updated.emit("처리 완료!")
                self.processing_complete.emit(primary_report_path, secondary_report_path)

        except Exception as e:
            if self._is_running:  # Only emit if not stopped
                error_msg = f"오류 발생: {str(e)}\n{traceback.format_exc()}"
                self.log_message.emit(error_msg)
                self.processing_error.emit(error_msg)

    # Placeholder for potential callback mechanism
    # def update_progress(self, current, total):
    #      percentage = int((current / total) * 100) if total > 0 else 0
    #      self.progress_updated.emit(percentage)
    #      self.log_message.emit(f"진행률: {current}/{total}")

class MainWindow(QMainWindow):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        # Pass config to processor
        self.processor = Processor(config)
        self.input_file = None
        self.processing_thread = None
        self.auto_save_timer = None
        
        self.setWindowTitle("상품 가격 비교 시스템")
        self.setGeometry(100, 100, 
                        int(config['GUI']['WINDOW_WIDTH']),
                        int(config['GUI']['WINDOW_HEIGHT']))
        
        # Enable drag & drop
        self.setAcceptDrops(True)
        
        # Apply dark mode if enabled
        if config['GUI'].get('ENABLE_DARK_MODE', False):
            self.apply_dark_mode()
        
        self.init_ui()
        self.setup_auto_save()
    
    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Top section (File handling and controls)
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)
        
        # File handling group
        file_group = QGroupBox("파일 처리")
        file_layout = QVBoxLayout()
        
        # Drag & drop area with improved styling
        self.drop_label = QLabel("엑셀 파일을 여기에 드래그하거나, 버튼을 클릭하여 선택하세요.")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 30px;
                background: #f0f0f0;
                font-size: 14px;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background: #e8f5e9;
            }
        """)
        file_layout.addWidget(self.drop_label)

        # File selection button with icon
        self.select_button = QPushButton("엑셀 파일 선택...")
        self.select_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.select_button.clicked.connect(self.select_file_dialog)
        file_layout.addWidget(self.select_button)
        
        file_group.setLayout(file_layout)
        top_layout.addWidget(file_group)
        
        # Processing controls group
        controls_group = QGroupBox("처리 제어")
        controls_layout = QHBoxLayout()
        
        # Start button with icon
        self.start_button = QPushButton("분석 시작")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.on_start_clicked)
        controls_layout.addWidget(self.start_button)
        
        # Stop button with icon
        self.stop_button = QPushButton("중지")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        controls_layout.addWidget(self.stop_button)
        
        # Clear cache button with icon
        self.clear_cache_button = QPushButton("캐시 비우기")
        self.clear_cache_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.clear_cache_button.clicked.connect(self.clear_cache)
        controls_layout.addWidget(self.clear_cache_button)
        
        controls_group.setLayout(controls_layout)
        top_layout.addWidget(controls_group)
        
        # Progress section
        progress_group = QGroupBox("진행 상황")
        progress_layout = QVBoxLayout()
        
        # Progress bar with improved styling
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #aaa;
                border-radius: 3px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        progress_layout.addWidget(self.progress)
        
        # Status label with improved styling
        self.status_label = QLabel("대기 중... 파일을 선택하세요.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #666;
            }
        """)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        top_layout.addWidget(progress_group)
        
        # Add top section to splitter
        splitter.addWidget(top_section)
        
        # Bottom section (Log area)
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)
        
        # Log group with improved styling
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout()
        
        # Log area with improved styling
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QPlainTextEdit {
                font-family: 'Consolas', monospace;
                font-size: 12px;
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        self.log_area.setMaximumBlockCount(int(self.config['GUI']['MAX_LOG_LINES']))
        log_layout.addWidget(self.log_area)
        
        log_group.setLayout(log_layout)
        bottom_layout.addWidget(log_group)
        
        # Add bottom section to splitter
        splitter.addWidget(bottom_section)
        
        # Set initial splitter sizes (60% top, 40% bottom)
        splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])
        
        # Setup logging handler for GUI
        self._setup_gui_logging()

    def setup_auto_save(self):
        """Setup auto-save functionality"""
        if self.config['GUI'].get('AUTO_SAVE_INTERVAL', 300) > 0:
            from PyQt5.QtCore import QTimer
            self.auto_save_timer = QTimer()
            self.auto_save_timer.timeout.connect(self.auto_save)
            self.auto_save_timer.start(self.config['GUI']['AUTO_SAVE_INTERVAL'] * 1000)

    def _setup_gui_logging(self):
        """Setup GUI logging handler"""
        # Create and configure GUI log handler
        gui_handler = GUILogHandler(self.log_area)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(gui_handler)
        
        # Set log level based on config
        log_level = getattr(logging, self.config['GUI'].get('LOG_LEVEL', 'INFO').upper())
        root_logger.setLevel(log_level)

    def auto_save(self):
        """Auto-save current state"""
        if self.input_file and self.processing_thread and self.processing_thread.isRunning():
            try:
                # Save current progress
                self.save_progress()
            except Exception as e:
                self.log_message.emit(f"자동 저장 실패: {str(e)}")

    def save_progress(self):
        """Save current processing progress"""
        # Implement progress saving logic here
        pass

    def apply_dark_mode(self):
        """Apply dark mode theme"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        
        # Set dark mode stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #2a82da;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3292ea;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
            QGroupBox {
                color: white;
                border: 1px solid #666666;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QProgressBar {
                border: 1px solid #666666;
                border-radius: 3px;
                text-align: center;
                background-color: #353535;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
            }
            QPlainTextEdit {
                background-color: #252525;
                color: white;
                border: 1px solid #666666;
                border-radius: 3px;
            }
        """)

    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            url = mime_data.urls()[0]
            if url.isLocalFile() and url.toLocalFile().lower().endswith(('.xlsx', '.xls')):
                event.acceptProposedAction()
                self.drop_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #4CAF50; /* Green border */
                        border-radius: 5px;
                        padding: 30px;
                        background: #e8f5e9; /* Light green background */
                        font-size: 14px;
                    }
                """)
            else:
                event.ignore()
                self._reset_drop_label_style()
        else:
            event.ignore()
            self._reset_drop_label_style()
            
    def dragLeaveEvent(self, event):
        self._reset_drop_label_style()

    def dropEvent(self, event):
        self._reset_drop_label_style()
        url = event.mimeData().urls()[0]
        filepath = url.toLocalFile()
        self._handle_file_selected(filepath)

    def select_file_dialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, 
                    "엑셀 파일 선택", "",
                    "Excel Files (*.xlsx *.xls)", options=options)
        if fileName:
            self._handle_file_selected(fileName)

    def _handle_file_selected(self, filepath: str):
         self.input_file = filepath
         self.drop_label.setText(f"선택된 파일: {os.path.basename(self.input_file)}")
         self.status_label.setText("파일 선택됨. '분석 시작' 버튼을 누르세요.")
         self.start_button.setEnabled(True)
         self.progress.setValue(0)
         self.log_area.clear()
         logging.info(f"입력 파일 선택됨: {self.input_file}")

    def _reset_drop_label_style(self):
         self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 30px;
                background: #f0f0f0;
                font-size: 14px;
            }
        """)
    
    @pyqtSlot()
    def on_start_clicked(self):
        if not self.input_file or not os.path.exists(self.input_file):
            QMessageBox.warning(self, "파일 오류", "유효한 엑셀 파일을 선택해주세요.")
            return
            
        if self.processing_thread and self.processing_thread.isRunning():
             QMessageBox.information(self, "처리 중", "이미 분석 작업이 진행 중입니다.")
             return

        self.start_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.clear_cache_button.setEnabled(False)
        self.status_label.setText("처리 시작 중...")
        self.progress.setValue(0)
        self.log_area.clear()
        logging.info("분석 시작...")
        
        # Create and start the worker thread
        self.processing_thread = ProcessingThread(self.processor, self.input_file)
        
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.log_message.connect(self.append_log)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.finished.connect(self.on_thread_finished)
        
        self.processing_thread.start()

    @pyqtSlot(int)
    def update_progress(self, value):
        self.progress.setValue(value)

    @pyqtSlot(str)
    def update_status(self, message):
        self.status_label.setText(message)

    @pyqtSlot(str)
    def append_log(self, message):
        self.log_area.appendPlainText(message)

    @pyqtSlot(str, str)
    def on_processing_complete(self, primary_report_path, secondary_report_path):
        self.status_label.setText("분석 완료!")
        logging.info("분석 작업 완료.")
        
        # 결과 경로가 있는지 확인
        has_primary = primary_report_path and os.path.exists(primary_report_path)
        has_secondary = secondary_report_path and os.path.exists(secondary_report_path)
        
        if has_primary or has_secondary:
            result_message = "분석이 완료되었습니다.\n\n"
            
            if has_primary:
                result_message += f"1차 보고서: {os.path.basename(primary_report_path)}\n"
            
            if has_secondary:
                result_message += f"2차 보고서: {os.path.basename(secondary_report_path)}\n"
                
            result_message += f"\n결과 폴더: {os.path.dirname(primary_report_path or secondary_report_path)}"
            
            QMessageBox.information(self, "완료", result_message)
            
            # 결과 폴더 열기
            try:
                output_dir = os.path.dirname(primary_report_path or secondary_report_path)
                os.startfile(output_dir)
                logging.info(f"결과 폴더 열림: {output_dir}")
            except AttributeError: # os.startfile not available on all platforms
                logging.warning("os.startfile은 이 플랫폼에서 지원되지 않습니다.")
                pass
            except Exception as e:
                logging.warning(f"결과 폴더 열기 실패: {e}")
        else:
            QMessageBox.warning(self, "완료", "분석은 완료되었으나 보고서가 생성되지 않았습니다.")

    @pyqtSlot(str)
    def on_processing_error(self, error_message):
        self.status_label.setText("오류 발생!")
        QMessageBox.critical(self, "오류", f"처리 중 오류가 발생했습니다:\n{error_message}")

    @pyqtSlot()
    def on_thread_finished(self):
        self.start_button.setEnabled(True) # Re-enable based on file selection state
        self.select_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)
        self.processing_thread = None # Clear the thread reference
        # Reset progress if needed, or leave it at 100%
        # self.progress.setValue(0) 

    @pyqtSlot()
    def clear_cache(self):
         reply = QMessageBox.question(self, '캐시 비우기', 
             "정말로 캐시를 비우시겠습니까? 다음 실행 시 시간이 더 오래 걸릴 수 있습니다.",
             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

         if reply == QMessageBox.Yes:
             try:
                 self.processor.cache.clear()
                 logging.info("캐시가 성공적으로 비워졌습니다.")
                 QMessageBox.information(self, "성공", "캐시가 성공적으로 비워졌습니다.")
             except Exception as e:
                 error_msg = f"캐시 비우기 실패: {e}"
                 logging.error(error_msg)
                 QMessageBox.critical(self, "오류", error_msg)

    @pyqtSlot()
    def on_stop_clicked(self):
        """Handle stop button click"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 
                '처리 중지',
                "진행 중인 처리를 중지하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.stop_button.setEnabled(False)
                self.status_label.setText("처리 중지 중...")
                self.log_message.emit("사용자 요청으로 처리 중지 중...")

    def closeEvent(self, event):
        """Handle window close event"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 
                '종료 확인',
                "분석 작업이 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.log_message.emit("사용자 요청으로 분석 중단 및 종료 중...")
                self.processing_thread.stop()
                self.processing_thread.wait(2000)  # Wait max 2 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# Custom logging handler to emit logs to the GUI text area
class GUILogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.widget = text_widget
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        # Ensure this runs in the GUI thread if logs come from other threads
        # QMetaObject.invokeMethod(self.widget, "appendPlainText", Qt.QueuedConnection, Q_ARG(str, msg))
        # For simplicity, assuming logs primarily come from GUI or signals handle threading:
        self.widget.appendPlainText(msg) 