"""
데이터 전처리 및 파일 관련 유틸리티 함수 모음.

이 모듈은 로깅 설정, 입력 파일 유효성 검사, 데이터 정제 (예: 상품명 정리),
파일 분할 및 병합 등 RPA 프로세스의 다양한 단계에서 사용될 수 있는
전처리 및 파일 관리 함수들을 제공합니다.
"""
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Dict, List, Optional

import pandas as pd
# Unused imports removed
# from openpyxl import load_workbook
# from openpyxl.styles import PatternFill

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Optional[str] = None) -> None:
    """로깅 설정을 초기화하고 구성합니다.

    콘솔 핸들러(INFO 레벨, ERROR 레벨 분리)와 파일 핸들러(DEBUG 레벨 전체 로그,
    ERROR 레벨 오류 로그 분리)를 설정합니다. 로그 파일은 회전 및 시간 기반 회전을 지원합니다.
    처리되지 않은 예외를 로깅하는 핸들러도 추가합니다.

    Args:
        log_dir: 로그 파일을 저장할 디렉토리 경로. None이면 파일 로깅 비활성화.

    Raises:
        Exception: 로깅 설정 중 오류 발생 시.
    """
    try:
        # Create log directory if specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Get current timestamp for log file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n"
            "File: %(pathname)s:%(lineno)d\n"
            "Function: %(funcName)s\n"
            "%(message)s"
        )

        simple_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

        # Error console handler (ERROR level)
        error_console_handler = logging.StreamHandler(sys.stderr)
        error_console_handler.setLevel(logging.ERROR)
        error_console_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_console_handler)

        if log_dir:
            # File handler for all logs (DEBUG level)
            all_log_file = os.path.join(log_dir, f"all_{timestamp}.log")
            file_handler = RotatingFileHandler(
                all_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)

            # Error log file handler (ERROR level)
            error_log_file = os.path.join(log_dir, f"error_{timestamp}.log")
            error_file_handler = TimedRotatingFileHandler(
                error_log_file,
                when="midnight",
                interval=1,
                backupCount=30,  # Keep 30 days of error logs
                encoding="utf-8",
            )
            error_file_handler.setLevel(logging.ERROR)
            error_file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_file_handler)

            # Create a copy of the latest log file instead of symlink
            latest_log = os.path.join(log_dir, "latest.log")
            if os.path.exists(latest_log):
                os.remove(latest_log)
            shutil.copy2(all_log_file, latest_log)

        # Add unhandled exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            """처리되지 않은 예외를 로깅하는 핸들러.

            KeyboardInterrupt는 로깅하지 않고 기본 처리를 따릅니다.
            그 외 예외는 ERROR 레벨로 로깅합니다.
            """
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
            else:
                root_logger.error(
                    "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
                )

        sys.excepthook = handle_exception

        # Log successful setup
        root_logger.info("Logging system initialized successfully")

    except Exception as e:
        # If logging setup fails, try to log to console as last resort
        print(f"Failed to setup logging: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise


class LogCapture:
    """특정 코드 블록 실행 동안 로그 메시지를 캡처하는 컨텍스트 관리자.

    지정된 로거에 임시 핸들러를 추가하여 해당 로거를 통해 생성되는 로그 레코드를
    내부 리스트에 저장합니다. 컨텍스트 블록을 벗어나면 핸들러는 자동으로 제거됩니다.
    테스트 또는 특정 작업의 로깅 출력을 분리하여 확인하는 데 유용합니다.

    Attributes:
        logger_name (str): 캡처할 대상 로거의 이름.
        level (int): 캡처할 로그 레벨 (기본값: logging.DEBUG).
        records (list): 캡처된 logging.LogRecord 객체 리스트.

    Examples:
        >>> logger = logging.getLogger('test_logger')
        >>> logger.addHandler(logging.StreamHandler())
        >>> logger.setLevel(logging.INFO)
        >>> with LogCapture('test_logger') as lc:
        ...     logger.info("Info message")
        ...     logger.debug("Debug message - not captured by default") # Level < INFO
        >>> messages = lc.get_messages()
        >>> print(messages)
        ['Info message']
    """

    def __init__(self, logger_name: str, level: int = logging.DEBUG):
        """LogCapture 인스턴스를 초기화합니다.

        Args:
            logger_name: 캡처할 대상 로거의 이름.
            level: 캡처할 최소 로그 레벨.
        """
        self.logger_name = logger_name
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.handler = None
        self.records = []

    def __enter__(self):
        """컨텍스트 관리자 진입 시 로거에 임시 핸들러를 추가합니다."""
        class RecordListHandler(logging.Handler):
            """로그 레코드를 내부 리스트에 저장하는 간단한 핸들러."""
            def __init__(self, records):
                super().__init__()
                self.records = records

            def emit(self, record):
                """수신된 로그 레코드를 리스트에 추가합니다."""
                self.records.append(record)

        self.handler = RecordListHandler(self.records)
        self.handler.setLevel(self.level)
        self.logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료 시 로거에서 임시 핸들러를 제거합니다."""
        if self.handler:
            self.logger.removeHandler(self.handler)

    def get_records(self) -> List[logging.LogRecord]:
        """캡처된 원본 LogRecord 객체 리스트를 반환합니다.

        Returns:
            List[logging.LogRecord]: 캡처된 로그 레코드 객체 리스트.
        """
        return self.records

    def get_messages(self) -> List[str]:
        """캡처된 로그 레코드를 포맷팅된 문자열 메시지 리스트로 반환합니다.

        Returns:
            List[str]: 포맷팅된 로그 메시지 문자열 리스트.
        """
        return [self.handler.format(record) for record in self.records]


def validate_input_file(file_path: str, required_columns: List[str]) -> bool:
    """입력 파일의 존재 여부, 확장자, 필수 컬럼 포함 여부를 검증합니다.

    Args:
        file_path: 검증할 입력 Excel 파일 경로.
        required_columns: 파일에 반드시 포함되어야 하는 컬럼명 리스트.

    Returns:
        bool: 파일이 유효하면 True, 그렇지 않으면 False.
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logger.error("Input file not found: %s", file_path)
            return False

        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in [".xls", ".xlsx", ".xlsm"]:
            logger.error(
                "Invalid file format: %s. Expected Excel file (.xls, .xlsx, .xlsm)", ext
            )
            return False

        # Verify file can be opened and read headers
        try:
            df_headers = pd.read_excel(file_path, nrows=0)
        except FileNotFoundError:
            logger.error("Input file not found during read: %s", file_path)
            return False
        except pd.errors.EmptyDataError:
            logger.error("Input file is empty: %s", file_path)
            return False
        except (OSError, IOError) as e:
            logger.error("Failed to open or read header from Excel file: %s", e)
            return False

        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df_headers.columns]
        if missing_cols:
            logger.error("Missing required columns: %s", ', '.join(missing_cols))
            return False

        return True

    except (OSError, AttributeError, TypeError, ValueError, KeyError) as e:
        logger.error("Error validating required columns: %s", e)
        return False

    # Catch-all for truly unexpected validation issues
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error("Unexpected error during input file validation: %s", e, exc_info=True)
        return False


def clean_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 '상품명' 컬럼을 지정된 규칙에 따라 정리합니다.

    - '//' 이후의 모든 문자열 제거
    - 숫자와 하이픈 조합 (예: 123-456) 등의 패턴 제거
    - 특정 불필요 키워드 제거 ("정품", "NEW", "특가", "주문제작" 관련 등)
    - 연속된 공백을 단일 공백으로 변경
    - 앞뒤 공백 제거

    Args:
        df: 처리할 pandas DataFrame. '상품명' 컬럼이 있어야 함.

    Returns:
        pd.DataFrame: 상품명이 정리된 DataFrame. '상품명' 컬럼이 없거나 처리 중 오류 발생 시 원본 DataFrame 반환.

    Raises:
        KeyError: '상품명' 컬럼이 DataFrame에 없을 경우 (내부에서 처리).
    """
    logger.info("Cleaning product names...")
    if "상품명" not in df.columns:
        logger.warning("'상품명' column not found in DataFrame. Skipping cleaning.")
        return df

    try:
        # Define patterns and keywords to remove
        comment_pattern = r"\s*//.*" # Pattern for comments starting with //
        noise_pattern = r"(\d{4}_[A-Z]\.)|(\d+\+\d+)|[^a-zA-Z0-9가-힣\s]"
        keywords_to_remove = [
            "정품", "NEW", "특가", "주문제작타올", "주문제작수건", "결혼답례품 수건",
            "답례품수건", "주문제작 수건", "돌답례품수건", "명절선물세트", "각종행사수건",
        ]

        # Apply cleaning steps
        # 1. Remove comments
        df["상품명"] = df["상품명"].astype(str).str.replace(comment_pattern, "", regex=True)
        # 2. Remove noise patterns
        df["상품명"] = df["상품명"].str.replace(noise_pattern, " ", regex=True)
        # 3. Remove specific keywords
        for keyword in keywords_to_remove:
            df["상품명"] = df["상품명"].str.replace(keyword, "", regex=False)
        # 4. Normalize whitespace and strip
        df["상품명"] = df["상품명"].str.replace(r"\s+", " ", regex=True).str.strip()

        logger.info("Product names cleaned successfully.")
        return df

    except (AttributeError, TypeError, ValueError) as e:
        logger.error("Error cleaning product names: %s", e)
        return df # Return original DataFrame on error


def process_input_file(
    file_path: str, config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """입력 Excel 파일을 읽고 유효성을 검사하며 기본적인 전처리를 수행합니다.

    수행 단계:
    1. 설정 파일에서 필수 컬럼 목록 로드.
    2. `validate_input_file`을 사용하여 파일 유효성 검사.
    3. Pandas를 사용하여 Excel 파일 읽기.
    4. 설정에 따라 `clean_product_names` 자동 실행.
    5. `normalize_column_types` 및 `handle_missing_values` 실행.
    6. 설정에 따라 중복 상품 코드(`상품Code`) 제거.

    Args:
        file_path: 처리할 입력 Excel 파일 경로.
        config: 처리 설정을 담고 있는 딕셔너리.

    Returns:
        Optional[pd.DataFrame]: 전처리된 DataFrame. 유효성 검사 실패 또는
                               처리 중 오류 발생 시 None을 반환합니다.
    """
    logger.info("Processing input file: %s", file_path)

    try:
        # Extract required columns from config
        required_columns = config["EXCEL"].get(
            "REQUIRED_COLUMNS",
            ["상품명", "판매단가(V포함)", "상품Code", "본사 이미지", "본사상품링크"],
        )

        # Validate input file
        if not validate_input_file(file_path, required_columns):
            return None

        # Read the Excel file
        df = pd.read_excel(file_path)

        # Automatically clean product names if enabled in config
        if config.get("PROCESSING", {}).get("AUTO_CLEAN_PRODUCT_NAMES", True):
            df = clean_product_names(df)
            logger.info("Product names cleaned automatically")

        # Add other preprocessing steps here
        df = normalize_column_types(df)
        df = handle_missing_values(df, required_columns)

        # Check for duplicate products
        if config.get("EXCEL", {}).get("ENABLE_DUPLICATE_DETECTION", True):
            if "상품Code" in df.columns:
                duplicates = df.duplicated(subset=["상품Code"], keep="first")
                if duplicates.any():
                    logger.warning("Found %d duplicate product codes", duplicates.sum())
                    # Keep first occurrence
                    df = df[~duplicates]
            else:
                logger.warning("'상품Code' column not found, skipping duplicate detection.")

        return df

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, OSError, IOError) as e:
        logger.error(f"Error reading input Excel file '{file_path}': {e}")
        return None
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error(f"Unexpected error processing input file '{file_path}': {e}", exc_info=True)
        return None


def normalize_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 주요 컬럼들의 데이터 타입을 표준화합니다.

    - 특정 문자열 컬럼들을 `str` 타입으로 변환하고 `nan` 문자열을 빈 문자열로 바꿉니다.
    - 특정 숫자 컬럼들을 `float` 타입으로 변환하고 변환 실패 시 0으로 채웁니다.

    Args:
        df: 타입을 정규화할 입력 DataFrame.

    Returns:
        pd.DataFrame: 컬럼 타입이 정규화된 DataFrame. 오류 발생 시 원본 DataFrame 반환.
    """
    try:
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()

        # Normalize string columns
        string_cols = [
            "상품명",
            "담당자",
            "업체명",
            "업체코드",
            "중분류카테고리",
            "상품Code",
            "본사상품링크",
            "구분",
        ]
        for col in string_cols:
            if col in normalized_df.columns:
                # Ensure column is actually string before replace
                if not pd.api.types.is_string_dtype(normalized_df[col]):
                    normalized_df[col] = normalized_df[col].astype(str)
                normalized_df[col] = normalized_df[col].replace("nan", "")

        # Ensure price/numeric columns are floats
        numeric_cols = ["판매단가(V포함)", "기본수량(1)"]
        for col in numeric_cols:
            if col in normalized_df.columns:
                normalized_df[col] = pd.to_numeric(normalized_df[col], errors="coerce")
                # Fill NaN resulting from coercion with 0
                normalized_df[col] = normalized_df[col].fillna(0)

        # Normalize image columns to string (they might be formulas later)
        normalized_df["본사 이미지"] = normalized_df["본사 이미지"].astype(str)
        normalized_df["고려기프트 이미지"] = normalized_df["고려기프트 이미지"].astype(str)
        normalized_df["네이버 이미지"] = normalized_df["네이버 이미지"].astype(str)

        logger.info("Column types normalized successfully.")
        return normalized_df

    except (AttributeError, TypeError, ValueError, KeyError) as e:
        logger.error(f"Error normalizing column types: {e}")
        return df # Return potentially partially processed DataFrame
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error(f"Unexpected error normalizing column types: {e}", exc_info=True)
        return df


def handle_missing_values(
    df: pd.DataFrame, required_columns: List[str]
) -> pd.DataFrame:
    """DataFrame의 결측치를 처리하고 특정 컬럼의 기본값을 설정합니다.

    - 문자열 컬럼의 NaN은 빈 문자열('')로 채웁니다.
    - 숫자 컬럼의 NaN은 0으로 채웁니다.
    - 필수 컬럼이 누락된 경우 경고를 로깅하고 빈 값으로 추가합니다.
    - '구분' 컬럼의 값이 'A' 또는 'P'가 아니거나 NaN인 경우 'A'로 설정합니다.

    Args:
        df: 결측치를 처리할 입력 DataFrame.
        required_columns: 반드시 존재해야 하는 컬럼 목록.

    Returns:
        pd.DataFrame: 결측치가 처리된 DataFrame. 오류 발생 시 원본 DataFrame 반환.
    """
    try:
        # Create a copy to avoid modifying the original
        handled_df = df.copy()

        # Define column types for appropriate filling
        string_like_cols = [
            "상품명", "담당자", "업체명", "업체코드", "중분류카테고리",
            "상품Code", "본사상품링크", "구분"
        ]
        numeric_like_cols = ["판매단가(V포함)", "기본수량(1)"]

        # Fill missing values based on type
        for col in handled_df.columns:
            if col in string_like_cols:
                handled_df[col] = handled_df[col].fillna("")
            elif col in numeric_like_cols:
                handled_df[col] = handled_df[col].fillna(0)
            # Add handling for other types if necessary

        # Ensure required columns (after potential fillna) exist and handle if not
        # Note: normalize_column_types should ideally ensure these exist
        # This is an extra safety check
        for col in required_columns:
            if col not in handled_df.columns:
                logger.warning("Required column '%s' still missing after handling NaNs. Adding empty.", col)
                if col in string_like_cols:
                    handled_df[col] = ""
                elif col in numeric_like_cols:
                    handled_df[col] = 0
                # Add default for other types if needed

        # Special handling for '구분' column
        if "구분" in handled_df.columns:
            # Default to 'A' (approval management) if invalid or missing
            handled_df["구분"] = handled_df["구분"].apply(
                lambda x: "A" if not isinstance(x, str) or x not in ["A", "P"] else x
            ).astype(str) # Ensure it's string type

        logger.info("Missing value handling complete.")
        return handled_df

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Error handling missing values: {e}")
        return df # Return DataFrame as is on error
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error(f"Unexpected error handling missing values: {e}", exc_info=True)
        return df


def split_large_file(
    file_path: str, threshold: int = 300, clean_names: bool = True
) -> List[str]:
    """주어진 Excel 파일이 지정된 임계값(행 수)보다 크면 여러 개의 작은 파일로 분할합니다.

    분할된 파일들은 원본 파일명에 `_part_1`, `_part_2` 등의 접미사가 붙어 저장됩니다.
    분할 전에 상품명 정제 옵션을 선택할 수 있습니다.

    Args:
        file_path: 분할할 원본 Excel 파일 경로.
        threshold: 분할 기준이 되는 최대 행 수 (기본값: 300).
        clean_names: 분할 전 상품명 정제 실행 여부 (기본값: True).

    Returns:
        List[str]: 분할된 파일들의 경로 리스트. 분할이 필요 없거나 오류 발생 시
                   원본 파일 경로만 포함하는 리스트 반환.

    Raises:
        FileNotFoundError: 입력 파일이 존재하지 않을 경우.
        ValueError: 임계값이 0 이하일 경우.
    """
    logger.info("Checking if file needs splitting: %s (Threshold: %d)", file_path, threshold)

    if threshold <= 0:
        raise ValueError("Split threshold must be positive.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    output_files = []
    try:
        # Read the entire Excel file to check the number of rows
        df = pd.read_excel(file_path)
        num_rows = len(df)

        if num_rows <= threshold:
            logger.info("File size (%d rows) is within the threshold. No splitting needed.", num_rows)
            return [file_path] # Return original file path in a list

        logger.info("File size (%d rows) exceeds threshold (%d). Splitting file...", num_rows, threshold)

        # Clean product names before splitting if requested
        if clean_names:
            try:
                df = clean_product_names(df)
            except (KeyError, AttributeError, TypeError, ValueError) as clean_err:
                logger.warning("Failed to clean product names before splitting: %s. Proceeding without cleaning.", clean_err)

        # Split the DataFrame
        num_parts = (num_rows + threshold - 1) // threshold # Ceiling division
        directory, filename = os.path.split(file_path)
        name, ext = os.path.splitext(filename)

        for i in range(num_parts):
            start_row = i * threshold
            end_row = min((i + 1) * threshold, num_rows)
            part_df = df.iloc[start_row:end_row]

            part_filename = f"{name}_part_{i+1}{ext}"
            part_filepath = os.path.join(directory, part_filename)

            # Save the part
            part_df.to_excel(part_filepath, index=False)
            output_files.append(part_filepath)
            logger.info("Created split file: %s (%d rows)", part_filepath, len(part_df))

        logger.info("Successfully split file into %d parts.", num_parts)
        return output_files

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, OSError, IOError, AttributeError, KeyError) as e:
        logger.error(f"Error splitting file '{file_path}': {e}")
        return [file_path] # Return original path in list on error
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error(f"Unexpected error splitting file '{file_path}': {e}", exc_info=True)
        return [file_path]


def merge_result_files(file_paths: List[str], original_input: str) -> Optional[str]:
    """분할 처리된 결과 파일들을 다시 하나의 파일로 병합합니다.

    병합된 파일은 원본 입력 파일명에 `_merged_{timestamp}` 접미사가 붙어 저장됩니다.
    병합 과정에서 각 파일의 헤더는 첫 번째 파일을 제외하고 무시됩니다.

    Args:
        file_paths: 병합할 결과 파일들의 경로 리스트.
        original_input: 분할 전 원본 입력 파일 경로 (출력 파일명 생성에 사용).

    Returns:
        Optional[str]: 병합된 최종 파일 경로. 오류 발생 시 None 반환.

    Raises:
        ValueError: 입력 파일 리스트가 비어있을 경우.
    """
    if not file_paths:
        raise ValueError("File paths list cannot be empty for merging.")

    logger.info("Merging %d result files...", len(file_paths))

    merged_df = pd.DataFrame()
    all_dfs = []

    try:
        # Read all result files
        for _, fpath in enumerate(file_paths):
            if not os.path.exists(fpath):
                logger.warning("Result file not found, skipping: %s", fpath)
                continue
            try:
                df_part = pd.read_excel(fpath)
                # Assume first file's header is the correct one
                # For subsequent files, skip header row if reading directly
                # However, reading all and concatenating is often simpler
                all_dfs.append(df_part)
                logger.debug("Read result file: %s (%d rows)", fpath, len(df_part))
            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, OSError, IOError) as read_err:
                logger.error("Error reading result file '%s': %s. Skipping.", fpath, read_err)
                continue

        if not all_dfs:
            logger.error("No valid result files could be read. Merging aborted.")
            return None

        # Concatenate DataFrames
        merged_df = pd.concat(all_dfs, ignore_index=True)
        logger.info("Concatenated %d DataFrames. Total rows: %d", len(all_dfs), len(merged_df))

        # Create output filename based on the original input file
        directory = os.path.dirname(original_input)
        base_name = os.path.splitext(os.path.basename(original_input))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_filename = f"{base_name}_merged_{timestamp}.xlsx"
        merged_filepath = os.path.join(directory, merged_filename)

        # Save the merged DataFrame
        merged_df.to_excel(merged_filepath, index=False)
        logger.info("Merged result file saved to: %s", merged_filepath)

        return merged_filepath

    except (MemoryError, ValueError, TypeError, KeyError, pd.errors.InvalidIndexError) as merge_err:
        logger.error(f"Error merging DataFrames: {merge_err}")
        return None
    except (OSError, IOError) as save_err:
        logger.error(f"Error saving merged file: {save_err}")
        return None
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error(f"Unexpected error merging files: {e}", exc_info=True)
        return None
