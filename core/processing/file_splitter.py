import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


class FileSplitter:
    """엑셀 파일 분할 및 병합을 담당하는 클래스"""

    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        파일 분할기 초기화

        Args:
            config: 애플리케이션 설정
            logger: 로깅 인스턴스
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # 설정에서 분할 관련 옵션 추출
        processing_config = config.get("PROCESSING", {})
        self.auto_split_files = processing_config.get("AUTO_SPLIT_FILES", True)
        self.split_threshold = processing_config.get("SPLIT_THRESHOLD", 300)
        self.auto_merge_results = processing_config.get("AUTO_MERGE_RESULTS", True)

    def needs_splitting(self, df: pd.DataFrame) -> bool:
        """
        데이터프레임이 분할이 필요한지 확인

        Args:
            df: 판다스 DataFrame

        Returns:
            분할 필요 여부
        """
        return self.auto_split_files and len(df) > self.split_threshold

    def split_input_file(self, df: pd.DataFrame, input_file: str) -> List[str]:
        """
        입력 파일을 분할하여 저장

        Args:
            df: 원본 데이터프레임
            input_file: 원본 파일 경로

        Returns:
            분할된 파일 경로 리스트
        """
        try:
            if not self.needs_splitting(df):
                self.logger.info(
                    f"No need to split file with {len(df)} rows (threshold: {self.split_threshold})"
                )
                return [input_file]

            base_name = os.path.splitext(input_file)[0]
            extension = os.path.splitext(input_file)[1]

            # 처리 유형 결정 ('A': 승인관리, 'P': 가격관리)
            processing_type = "승인관리"
            if "구분" in df.columns and not df["구분"].empty:
                if df["구분"].iloc[0].upper() == "P":
                    processing_type = "가격관리"

            # 현재 날짜
            current_date = datetime.now().strftime("%Y%m%d")

            # 분할 파일 수 계산
            total_rows = len(df)
            num_files = (total_rows + self.split_threshold - 1) // self.split_threshold

            self.logger.info(
                f"Splitting file into {num_files} parts (total rows: {total_rows})"
            )

            split_files = []
            for i in range(num_files):
                start_idx = i * self.split_threshold
                end_idx = min((i + 1) * self.split_threshold, total_rows)
                chunk_size = end_idx - start_idx

                # 승인관리1(300)-20230401.xlsx 형식으로 파일명 생성
                file_count = i + 1
                split_filename = f"{processing_type}{file_count}({chunk_size})-{current_date}{extension}"
                split_path = os.path.join(os.path.dirname(input_file), split_filename)

                # 청크 저장
                chunk_df = df.iloc[start_idx:end_idx].copy()
                chunk_df.to_excel(split_path, index=False)
                split_files.append(split_path)

                self.logger.info(
                    f"Created split file: {split_path} with {chunk_size} rows"
                )

            return split_files

        except Exception as e:
            self.logger.error(f"Error splitting file: {str(e)}", exc_info=True)
            return [input_file]  # 오류 발생 시 원본 파일 경로 반환

    def merge_result_files(self, result_files: List[str], original_input: str) -> str:
        """
        처리 결과 파일들을 하나로 병합

        Args:
            result_files: 처리 결과 파일 경로 리스트
            original_input: 원본 입력 파일 경로

        Returns:
            병합된 결과 파일 경로
        """
        try:
            if not self.auto_merge_results or len(result_files) <= 1:
                return result_files[0] if result_files else ""

            self.logger.info(f"Merging {len(result_files)} result files")

            # 각 파일 읽기
            dfs = []
            for file in result_files:
                try:
                    df = pd.read_excel(file)
                    if not df.empty:
                        dfs.append(df)
                        self.logger.debug(f"Read {len(df)} rows from {file}")
                    else:
                        self.logger.warning(f"Empty file: {file}")
                except Exception as e:
                    self.logger.error(f"Error reading file {file}: {str(e)}")

            if not dfs:
                self.logger.error("No valid files to merge")
                return result_files[0] if result_files else ""

            # 병합
            merged_df = pd.concat(dfs, ignore_index=True)

            # 병합 파일명 생성
            base_name = os.path.splitext(original_input)[0]
            merged_filename = f"{base_name}-merged-result.xlsx"

            # 병합 파일 저장
            merged_df.to_excel(merged_filename, index=False)

            self.logger.info(
                f"Created merged file: {merged_filename} with {len(merged_df)} rows"
            )

            return merged_filename

        except Exception as e:
            self.logger.error(f"Error merging result files: {str(e)}", exc_info=True)
            return result_files[0] if result_files else ""
