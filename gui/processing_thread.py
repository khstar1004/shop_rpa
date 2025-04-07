from PyQt5.QtCore import QThread, pyqtSignal
import traceback
import logging
import pandas as pd


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)       # Current progress percentage
    status_updated = pyqtSignal(str)          # Status messages
    log_message = pyqtSignal(str)             # Logging messages
    processing_complete = pyqtSignal(str, str)  # Primary and secondary report paths
    processing_error = pyqtSignal(str)        # Error message
    memory_usage_updated = pyqtSignal(float)  # Memory usage updates

    def __init__(self, processor, input_file):
        super().__init__()
        self.processor = processor
        self.input_file = input_file
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.status_updated.emit("파일 로딩 중...")
            try:
                df = pd.read_excel(self.input_file)
                total_items = len(df)
            except Exception as e:
                self.log_message.emit(f"입력 파일 로딩 오류: {e}")
                total_items = 0
                raise e

            self.status_updated.emit(f"{total_items}개 상품 처리 시작...")
            self.progress_updated.emit(0)

            primary_report_path, secondary_report_path = self.processor.process_file(self.input_file)

            if self._is_running:
                self.progress_updated.emit(100)
                self.status_updated.emit("처리 완료!")
                self.processing_complete.emit(primary_report_path, secondary_report_path)
        except Exception as e:
            if self._is_running:
                error_msg = f"오류 발생: {str(e)}\n{traceback.format_exc()}"
                self.log_message.emit(error_msg)
                self.processing_error.emit(error_msg) 