import logging
import os
import traceback
from typing import List, Optional

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

# Import ProductProcessor for type hinting (assuming it's accessible)
# Adjust the import path based on your project structure
from core.processing.main_processor import ProductProcessor
from .main_window import GUILogHandler


class ProcessingThread(QThread):
    """상품 처리를 위한 스레드"""

    # --- Refined Signals ---
    progress_updated = pyqtSignal(int, int)  # Emits (current_item_count, total_items)
    status_updated = pyqtSignal(str)  # Emits status messages for the UI
    error_occurred = pyqtSignal(str)  # Emits error messages
    processing_completed = pyqtSignal(list)  # Emits list of output file paths on success
    processing_stopped = pyqtSignal()  # Emits when processing is stopped prematurely
    log_message = pyqtSignal(str)  # Emits log messages
    processing_complete = pyqtSignal(str, str)  # Emits primary_path, secondary_path
    processing_error = pyqtSignal(str)  # Add processing_error signal

    def __init__(
        self,
        processor: ProductProcessor,
        input_files: List[str],
        output_dir: Optional[str] = None,
        limit: Optional[int] = None,
        product_limit: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.processor = processor
        self.input_files = input_files if isinstance(input_files, list) else [input_files]
        self.output_dir = output_dir
        self.limit = limit or product_limit
        self._is_running = True
        self.logger = logging.getLogger(__name__)
        
        # Set up logging handler to emit log messages
        self.log_handler = GUILogHandler(signal=self.log_message)
        self.logger.addHandler(self.log_handler)

        # --- Set up progress callback ---
        if hasattr(self.processor, "progress_callback"):
            self.processor.progress_callback = self._update_progress
        else:
            self.logger.warning("Processor object does not have 'progress_callback' attribute.")

    def __del__(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'log_handler'):
                self.logger.removeHandler(self.log_handler)
                self.log_handler = None
        except Exception as e:
            logging.error(f"Error in ProcessingThread cleanup: {str(e)}")

    def run(self):
        """스레드 실행 로직"""
        self._is_running = True
        output_files = []
        total_files = len(self.input_files)
        processed_files_count = 0

        try:
            self.status_updated.emit(f"처리 시작: {total_files}개 파일")
            self.logger.info(f"Starting processing of {total_files} files")

            for input_file in self.input_files:
                if not self._is_running:
                    self.logger.info("Processing stopped by user before processing next file.")
                    self.processing_stopped.emit()
                    return

                file_name = os.path.basename(input_file)
                self.status_updated.emit(f"파일 처리 중 ({processed_files_count + 1}/{total_files}): {file_name}...")
                self.logger.info(f"Processing file: {file_name}")
                
                if hasattr(self.processor, "_is_running"):
                    self.processor._is_running = True 
                
                result_file = None
                error_message = None
                try:
                    if self.limit is not None and self.limit > 0:
                        self.logger.info(f"Processing limited file: {file_name} (limit: {self.limit})")
                        if hasattr(self.processor, "_process_limited_file"):
                            result_file = self.processor._process_limited_file(
                                input_file, self.output_dir, self.limit
                            )
                        else:
                            error_message = f"Processor missing '_process_limited_file' method for file {file_name}"
                            self.logger.error(error_message)
                            self.processing_error.emit(error_message)
                    else:
                        self.logger.info(f"Processing full file: {file_name}")
                        if hasattr(self.processor, "_process_single_file"):
                            process_output = self.processor._process_single_file(
                                input_file, self.output_dir
                            )
                            if isinstance(process_output, tuple) and len(process_output) == 2:
                                result_file, error_message = process_output
                                if error_message:
                                    self.processing_error.emit(error_message)
                            elif isinstance(process_output, str):
                                result_file = process_output
                        else:
                            error_message = f"Processor missing '_process_single_file' method for file {file_name}"
                            self.logger.error(error_message)
                            self.processing_error.emit(error_message)

                    if result_file:
                        self.logger.info(f"Successfully processed '{file_name}' -> '{result_file}'")
                        output_files.append(result_file)
                        processed_files_count += 1
                        self.processing_complete.emit(result_file, "")
                    elif error_message:
                        self.logger.error(f"Error processing file '{file_name}': {error_message}")
                        self.error_occurred.emit(f"파일 '{file_name}' 처리 오류: {error_message}")
                        self.processing_error.emit(error_message)
                    else:
                        if self._is_running:
                            self.logger.warning(f"No output generated for file '{file_name}', possibly stopped or empty.")

                except Exception as e:
                    error_msg = f"파일 '{file_name}' 처리 중 예외 발생: {str(e)}\n{traceback.format_exc()}"
                    self.logger.error(error_msg)
                    if self._is_running:
                        self.error_occurred.emit(f"파일 '{file_name}' 처리 오류: {str(e)}")
                        self.processing_error.emit(str(e))

            if self._is_running:
                if output_files:
                    completion_msg = f"모든 처리 완료: {len(output_files)}개 파일 생성됨"
                    self.status_updated.emit(completion_msg)
                    self.logger.info(completion_msg)
                    self.processing_completed.emit(output_files)
                else:
                    completion_msg = "처리 완료 (생성된 파일 없음)"
                    self.status_updated.emit(completion_msg)
                    self.logger.info(completion_msg)
                    self.processing_completed.emit([])
            else:
                self.logger.info("Processing loop finished due to stop request.")

        except Exception as e:
            error_msg = f"전체 처리 중 예외 발생: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            if self._is_running:
                self.error_occurred.emit(f"처리 중 예상치 못한 오류: {str(e)}")
                self.processing_error.emit(str(e))

        finally:
            self._is_running = False
            if hasattr(self.processor, "progress_callback") and self.processor.progress_callback == self._update_progress:
                self.processor.progress_callback = None
            self.logger.info("Processing thread finished.")


    def stop(self):
        """처리 중단 요청"""
        if self._is_running:
            self.logger.info("Stop requested by user.")
            self._is_running = False
            # Signal the processor to stop its internal operations
            if hasattr(self.processor, "stop_processing"):
                 try:
                     self.processor.stop_processing()
                 except Exception as e:
                     self.logger.error(f"Error calling processor.stop_processing(): {e}")
            else:
                 self.logger.warning("Processor does not have 'stop_processing' method.")
            # Don't emit processing_stopped here, let the run loop handle it when it terminates


    def _update_progress(self, current: int, total: int):
        """프로세서로부터 진행 상황을 받아 UI 업데이트 시그널 발생"""
        # Check if the thread is still supposed to be running before emitting signals
        if self._is_running:
            try:
                self.progress_updated.emit(current, total)
                # Update status message less frequently or based on percentage?
                # Example: update every 10 items or at the end
                if total > 0 and (current % 10 == 0 or current == total or current == 0):
                     percentage = int((current / total) * 100)
                     self.status_updated.emit(f"처리 중... {current}/{total} ({percentage}%)")
                elif total == 0:
                     self.status_updated.emit(f"처리 중... (항목 없음)")

            except Exception as e:
                 # Catch Qt signal emission errors (less likely but possible)
                 self.logger.error(f"Error emitting progress signals: {e}")

# --- Remove old methods if they are no longer needed ---
# def _process_single_file(self, input_file): ...
# def _process_multiple_files(self): ...
# Remove memory usage signal if not implemented
# memory_usage_updated = pyqtSignal(float)
# Remove old signals if replaced
# log_message = pyqtSignal(str)
# processing_complete = pyqtSignal(str, str)
# processing_error = pyqtSignal(str)
# processing_finished = pyqtSignal(list) # Replaced by processing_completed
