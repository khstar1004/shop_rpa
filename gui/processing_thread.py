import logging
import os
import traceback
from typing import List, Optional

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

# Import ProductProcessor for type hinting (assuming it's accessible)
# Adjust the import path based on your project structure
from core.processing.main_processor import ProductProcessor


class ProcessingThread(QThread):
    """상품 처리를 위한 스레드"""

    # --- Refined Signals ---
    progress_updated = pyqtSignal(int, int)  # Emits (current_item_count, total_items)
    status_updated = pyqtSignal(str)  # Emits status messages for the UI
    error_occurred = pyqtSignal(str)  # Emits error messages
    processing_completed = pyqtSignal(list)  # Emits list of output file paths on success
    processing_stopped = pyqtSignal()  # Emits when processing is stopped prematurely

    def __init__(
        self,
        processor: ProductProcessor,
        input_files: List[str],
        output_dir: Optional[str] = None,
        limit: Optional[int] = None,
        product_limit: Optional[int] = None,
        parent=None, # Keep parent for QObject hierarchy
    ):
        super().__init__(parent) # Pass parent to superclass
        self.processor = processor
        # Ensure input_files is always a list
        self.input_files = (
            input_files if isinstance(input_files, list) else [input_files]
        )
        self.output_dir = output_dir
        self.limit = limit or product_limit  # Use either limit or product_limit
        self._is_running = True
        self.logger = logging.getLogger(__name__)

        # --- Set up progress callback ---
        # Ensure processor has the attribute before assigning
        if hasattr(self.processor, "progress_callback"):
            self.processor.progress_callback = self._update_progress
        else:
            self.logger.warning("Processor object does not have 'progress_callback' attribute.")


    def run(self):
        """스레드 실행 로직"""
        self._is_running = True # Ensure flag is true when starting
        output_files = []
        total_files = len(self.input_files)
        processed_files_count = 0

        try:
            self.status_updated.emit(f"처리 시작: {total_files}개 파일")

            for input_file in self.input_files:
                if not self._is_running:
                    self.logger.info("Processing stopped by user before processing next file.")
                    self.processing_stopped.emit()
                    return # Exit run method

                file_name = os.path.basename(input_file)
                self.status_updated.emit(f"파일 처리 중 ({processed_files_count + 1}/{total_files}): {file_name}...")
                
                # Reset processor's running flag for the new file
                if hasattr(self.processor, "_is_running"):
                    self.processor._is_running = True 
                
                result_file = None
                error_message = None
                try:
                    # Call the appropriate processor method based on limit
                    if self.limit is not None and self.limit > 0:
                         self.logger.info(f"Processing limited file: {file_name} (limit: {self.limit})")
                         # Use _process_limited_file (ensure it exists and handles output_dir)
                         if hasattr(self.processor, "_process_limited_file"):
                             result_file = self.processor._process_limited_file(
                                 input_file, self.output_dir, self.limit
                             )
                         else:
                              error_message = f"Processor missing '_process_limited_file' method for file {file_name}"
                              self.logger.error(error_message)
                    
                    else:
                         self.logger.info(f"Processing full file: {file_name}")
                         # Use _process_single_file (ensure it exists and handles output_dir)
                         if hasattr(self.processor, "_process_single_file"):
                             # _process_single_file might return (path, error_msg) or just path
                             process_output = self.processor._process_single_file(
                                 input_file, self.output_dir
                             )
                             # Handle different return types gracefully
                             if isinstance(process_output, tuple) and len(process_output) == 2:
                                 result_file, error_message = process_output
                             elif isinstance(process_output, str):
                                 result_file = process_output
                             # If process_output is None, result_file remains None

                         else:
                              error_message = f"Processor missing '_process_single_file' method for file {file_name}"
                              self.logger.error(error_message)

                    # Check processor's running flag after processing the file
                    # If processor stopped itself (e.g., internal error), reflect it
                    if hasattr(self.processor, "_is_running") and not self.processor._is_running:
                        if self._is_running: # Only log/emit stop if thread wasn't already stopped
                           self.logger.warning(f"Processor indicated stop during file: {file_name}")
                           # Don't emit processing_stopped here, let the main loop handle it or final state
                           # self.processing_stopped.emit() 
                           # self._is_running = False # Optionally stop the thread too

                    if result_file:
                        self.logger.info(f"Successfully processed '{file_name}' -> '{result_file}'")
                        output_files.append(result_file)
                        processed_files_count += 1
                    elif error_message:
                         self.logger.error(f"Error processing file '{file_name}': {error_message}")
                         self.error_occurred.emit(f"파일 '{file_name}' 처리 오류: {error_message}")
                    else:
                         # No result file and no specific error message from processor (could be stopped or empty)
                         if self._is_running: # Don't log as error if stopped deliberately
                             self.logger.warning(f"No output generated for file '{file_name}', possibly stopped or empty.")
                         # Do not increment processed_files_count here

                except Exception as e:
                    # Catch errors during the call to processor methods
                    error_msg = f"파일 '{file_name}' 처리 중 예외 발생: {str(e)}\n{traceback.format_exc()}"
                    self.logger.error(error_msg)
                    if self._is_running: # Only emit error if not stopped
                        self.error_occurred.emit(f"파일 '{file_name}' 처리 오류: {str(e)}")
                    # Continue to the next file even if one fails

            # --- Processing Finished ---
            if self._is_running:
                if output_files:
                    self.status_updated.emit(f"모든 처리 완료: {len(output_files)}개 파일 생성됨")
                    self.processing_completed.emit(output_files)
                else:
                    self.status_updated.emit("처리 완료 (생성된 파일 없음)")
                    self.processing_completed.emit([]) # Emit empty list for consistency
            else:
                 # If stopped before completing all files
                 self.logger.info("Processing loop finished due to stop request.")
                 # processing_stopped should have been emitted earlier

        except Exception as e:
            # Catch errors in the thread's main loop (outside file iteration)
            error_msg = f"전체 처리 중 예외 발생: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            if self._is_running: # Check run state before emitting
                self.error_occurred.emit(f"처리 중 예상치 못한 오류: {str(e)}")

        finally:
            self._is_running = False # Ensure flag is false on exit
            # Clean up processor callback?
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
