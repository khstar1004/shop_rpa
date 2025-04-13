import logging
import os
import traceback
from typing import List, Optional

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int, int)  # 진행 상황 (current_item, total_items)
    status_updated = pyqtSignal(str)  # 상태 메시지
    log_message = pyqtSignal(str)  # 로그 메시지
    processing_complete = pyqtSignal(str, str)  # 주 보고서와 보조 보고서 경로
    processing_error = pyqtSignal(str)  # 오류 메시지
    memory_usage_updated = pyqtSignal(float)  # 메모리 사용량 업데이트
    processing_finished = pyqtSignal(list)  # 처리 완료된 파일 목록

    def __init__(self, processor, input_files, output_dir=None, product_limit=None):
        super().__init__()
        self.processor = processor
        self.input_files = (
            input_files if isinstance(input_files, list) else [input_files]
        )
        self.output_dir = output_dir
        self.product_limit = product_limit
        self._is_running = True
        self.logger = logging.getLogger(__name__)

        # 진행 콜백 설정
        if hasattr(self.processor, "progress_callback"):
            self.processor.progress_callback = self.progress_updated.emit

    def stop(self):
        """처리 중단"""
        self._is_running = False
        if hasattr(self.processor, "progress_callback"):
            self.processor.progress_callback = None  # 중단 시 콜백 제거

        # Also signal stop to processor if it has a specific stop method
        if hasattr(self.processor, "stop_processing"):
            self.processor.stop_processing()

    def run(self):
        """스레드 실행"""
        try:
            if len(self.input_files) == 1:
                # 단일 파일 처리
                self._process_single_file(self.input_files[0])
            else:
                # 여러 파일 처리
                self._process_multiple_files()

        except Exception as e:
            if self._is_running:
                error_msg = f"오류 발생: {str(e)}\n{traceback.format_exc()}"
                self.log_message.emit(error_msg)
                self.processing_error.emit(error_msg)

    def _process_single_file(self, input_file):
        """단일 파일 처리"""
        try:
            self.status_updated.emit(f"파일 로딩 중: {os.path.basename(input_file)}...")
            try:
                df = pd.read_excel(input_file)
                total_items = len(df)
            except Exception as e:
                self.log_message.emit(f"입력 파일 로딩 오류: {e}")
                total_items = 0
                raise e

            self.status_updated.emit(f"{total_items}개 상품 처리 시작...")
            self.progress_updated.emit(0, total_items)  # Fix: Emit both current and total items

            if hasattr(self.processor, "process_file"):
                # 기존 API 호환성
                primary_report_path, secondary_report_path = (
                    self.processor.process_file(input_file)
                )
                if self._is_running:
                    self.progress_updated.emit(total_items, total_items)  # Final state
                    self.status_updated.emit("처리 완료!")
                    self.processing_complete.emit(
                        primary_report_path, secondary_report_path
                    )
            else:
                # 새 API 사용
                result_file = None
                if self.product_limit:
                    self.status_updated.emit(
                        f"최대 {self.product_limit}개 상품만 처리합니다..."
                    )
                    result_file = self.processor._process_limited_file(
                        input_file, self.output_dir, self.product_limit
                    )
                else:
                    # When no limit, need total items for progress
                    df = pd.read_excel(input_file)
                    total_items = len(df)
                    self.progress_updated.emit(0, total_items)
                    result_file = self.processor._process_single_file(
                        input_file, self.output_dir
                    )

                if self._is_running:
                    self.progress_updated.emit(total_items, total_items)  # Final state
                    if result_file:
                        self.status_updated.emit("처리 완료!")
                        self.processing_finished.emit([result_file])
                    else:
                        self.status_updated.emit("처리 실패!")
                        self.processing_error.emit("처리 중 오류가 발생했습니다.")

        except Exception as e:
            if self._is_running:
                error_msg = f"오류 발생: {str(e)}\n{traceback.format_exc()}"
                self.log_message.emit(error_msg)
                self.processing_error.emit(error_msg)

    def _process_multiple_files(self):
        """여러 파일 처리"""
        try:
            total_files = len(self.input_files)
            self.status_updated.emit(f"{total_files}개 파일 처리 시작...")
            
            # 초기 진행 상태 업데이트
            self.progress_updated.emit(0, total_files)
            
            # 각 파일마다 진행 상황을 추적하기 위한 변수
            processed_files = 0
            output_files = []
            
            # 파일별로 처리하면서 진행 상황 업데이트
            for idx, input_file in enumerate(self.input_files):
                if not self._is_running:
                    break
                
                file_name = os.path.basename(input_file)
                self.status_updated.emit(f"파일 처리 중 ({idx+1}/{total_files}): {file_name}")
                
                try:
                    # 개별 파일 처리
                    if hasattr(self.processor, "_process_single_file"):
                        result_file = self.processor._process_single_file(input_file, self.output_dir)
                        if result_file:
                            output_files.append(result_file)
                except Exception as e:
                    self.log_message.emit(f"파일 처리 오류 ({file_name}): {str(e)}")
                    continue
                
                # 진행 상황 업데이트
                processed_files += 1
                progress_percent = int((processed_files / total_files) * 100)
                self.progress_updated.emit(processed_files, total_files)
                
                # 처리 결과 로그
                self.log_message.emit(f"파일 처리 완료 ({idx+1}/{total_files}): {file_name}")
            
            # 최종 진행 상태 업데이트
            if self._is_running:
                self.progress_updated.emit(total_files, total_files)
                if output_files:
                    self.status_updated.emit(f"{len(output_files)}개 파일 처리 완료!")
                    self.processing_finished.emit(output_files)
                else:
                    self.status_updated.emit("처리 실패!")
                    self.processing_error.emit("처리 중 오류가 발생했습니다.")

        except Exception as e:
            if self._is_running:
                error_msg = f"오류 발생: {str(e)}\n{traceback.format_exc()}"
                self.log_message.emit(error_msg)
                self.processing_error.emit(error_msg)
