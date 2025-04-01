import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
import traceback
from typing import Optional, List, Dict, Any
import shutil
import pandas as pd
import numpy as np
import re
from openpyxl import load_workbook

logger = logging.getLogger(__name__)

def setup_logging(log_dir: Optional[str] = None) -> None:
    """Setup logging configuration with improved error handling and features."""
    try:
        # Create log directory if specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Get current timestamp for log file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'File: %(pathname)s:%(lineno)d\n'
            'Function: %(funcName)s\n'
            '%(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
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
            all_log_file = os.path.join(log_dir, f'all_{timestamp}.log')
            file_handler = RotatingFileHandler(
                all_log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file handler (ERROR level)
            error_log_file = os.path.join(log_dir, f'error_{timestamp}.log')
            error_file_handler = TimedRotatingFileHandler(
                error_log_file,
                when='midnight',
                interval=1,
                backupCount=30,  # Keep 30 days of error logs
                encoding='utf-8'
            )
            error_file_handler.setLevel(logging.ERROR)
            error_file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_file_handler)
            
            # Create a copy of the latest log file instead of symlink
            latest_log = os.path.join(log_dir, 'latest.log')
            if os.path.exists(latest_log):
                os.remove(latest_log)
            shutil.copy2(all_log_file, latest_log)
        
        # Add unhandled exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupts
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
            else:
                root_logger.error(
                    "Uncaught exception",
                    exc_info=(exc_type, exc_value, exc_traceback)
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
    """Context manager for capturing log output during a specific operation."""
    def __init__(self, logger_name: str, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.handler = None
        self.records = []
        
    def __enter__(self):
        class RecordListHandler(logging.Handler):
            def __init__(self, records):
                super().__init__()
                self.records = records
                
            def emit(self, record):
                self.records.append(record)
        
        self.handler = RecordListHandler(self.records)
        self.handler.setLevel(self.level)
        self.logger.addHandler(self.handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            self.logger.removeHandler(self.handler)
            
    def get_records(self) -> list:
        """Get captured log records."""
        return self.records
        
    def get_messages(self) -> list:
        """Get formatted log messages."""
        return [self.handler.format(record) for record in self.records]

def validate_input_file(file_path: str, required_columns: List[str]) -> bool:
    """
    Validate input file format and required columns.
    
    Args:
        file_path: Path to the input Excel file
        required_columns: List of required column names
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            return False
            
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in ['.xls', '.xlsx', '.xlsm']:
            logger.error(f"Invalid file format: {ext}. Expected Excel file (.xls, .xlsx, .xlsm)")
            return False
            
        # Verify file can be opened
        try:
            df = pd.read_excel(file_path, nrows=0)
        except Exception as e:
            logger.error(f"Failed to open Excel file: {e}")
            return False
            
        # Check for required columns
        for column in required_columns:
            if column not in df.columns:
                logger.error(f"Required column '{column}' not found in input file")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating input file: {e}")
        return False

def clean_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean product names according to manual requirements:
    - Remove '1-' prefix and numbers/hyphens before it
    - Remove special characters like '/' and '()'
    - Remove numbers in parentheses
    
    Args:
        df: Input DataFrame with '상품명' column
        
    Returns:
        DataFrame with cleaned product names
    """
    if '상품명' not in df.columns:
        logger.warning("Column '상품명' not found, skipping product name cleaning")
        return df
        
    logger.info("Cleaning product names")
    
    try:
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Remove '1-' and numbers/hyphens before it
        cleaned_df['상품명'] = cleaned_df['상품명'].str.replace(r'^\d+-', '', regex=True)
        
        # Remove special characters and brackets
        cleaned_df['상품명'] = cleaned_df['상품명'].str.replace(r'[/()]', '', regex=True)
        
        # Remove numbers in parentheses
        cleaned_df['상품명'] = cleaned_df['상품명'].str.replace(r'\(\d+\)', '', regex=True)
        
        # Trim extra whitespace
        cleaned_df['상품명'] = cleaned_df['상품명'].str.strip()
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning product names: {e}")
        return df  # Return original if error occurs

def process_input_file(file_path: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Process and validate input Excel file.
    
    Args:
        file_path: Path to the input Excel file
        config: Configuration dictionary
        
    Returns:
        DataFrame or None if validation fails
    """
    logger.info(f"Processing input file: {file_path}")
    
    # Extract required columns from config
    required_columns = config['EXCEL'].get('REQUIRED_COLUMNS', 
        ['상품명', '판매단가(V포함)', '상품Code', '본사 이미지', '본사상품링크'])
    
    # Validate input file
    if not validate_input_file(file_path, required_columns):
        return None
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Automatically clean product names if enabled in config
        if config['PROCESSING'].get('AUTO_CLEAN_PRODUCT_NAMES', True):
            df = clean_product_names(df)
            logger.info("Product names cleaned automatically")
        
        # Add other preprocessing steps here
        df = normalize_column_types(df)
        df = handle_missing_values(df, required_columns)
        
        # Check for duplicate products
        if config['EXCEL'].get('ENABLE_DUPLICATE_DETECTION', True):
            duplicates = df.duplicated(subset=['상품Code'], keep='first')
            if duplicates.any():
                logger.warning(f"Found {duplicates.sum()} duplicate product codes")
                # Keep first occurrence
                df = df[~duplicates]
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing input file: {e}")
        return None

def normalize_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column data types for consistent processing.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized data types
    """
    try:
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # Normalize string columns
        for col in ['상품명', '담당자', '업체명', '업체코드', '중분류카테고리', 
                    '상품Code', '본사상품링크', '구분']:
            if col in normalized_df.columns:
                normalized_df[col] = normalized_df[col].astype(str)
                normalized_df[col] = normalized_df[col].replace('nan', '')
        
        # Ensure price/numeric columns are floats
        for col in ['판매단가(V포함)', '기본수량(1)']:
            if col in normalized_df.columns:
                normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce')
                normalized_df[col] = normalized_df[col].fillna(0)
        
        return normalized_df
        
    except Exception as e:
        logger.error(f"Error normalizing column types: {e}")
        return df  # Return original if error occurs

def handle_missing_values(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        required_columns: List of columns that must have values
        
    Returns:
        DataFrame with missing values handled
    """
    try:
        # Create a copy to avoid modifying the original
        handled_df = df.copy()
        
        # Fill missing values for required columns
        for col in required_columns:
            if col in handled_df.columns:
                if col in ['상품명', '담당자', '업체명', '업체코드', '중분류카테고리', 
                           '상품Code', '본사상품링크', '구분']:
                    handled_df[col] = handled_df[col].fillna('')
                elif col in ['판매단가(V포함)', '기본수량(1)']:
                    handled_df[col] = handled_df[col].fillna(0)
        
        # Special handling for specific columns
        if '구분' in handled_df.columns:
            # Default to 'A' (approval management) if not specified
            handled_df['구분'] = handled_df['구분'].apply(
                lambda x: 'A' if (pd.isna(x) or x == '' or x not in ['A', 'P']) else x
            )
            
        return handled_df
        
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        return df  # Return original if error occurs

def split_large_file(file_path: str, threshold: int = 300, clean_names: bool = True) -> List[str]:
    """
    Split large Excel file into smaller chunks according to the manual requirements.
    
    Args:
        file_path: Path to the input Excel file
        threshold: Maximum rows per output file
        clean_names: Whether to clean product names
        
    Returns:
        List of paths to the split files
    """
    try:
        logger.info(f"Splitting file: {file_path} (threshold: {threshold})")
        
        # Read the input file
        df = pd.read_excel(file_path)
        
        # Clean product names if requested
        if clean_names:
            df = clean_product_names(df)
        
        # Get processing type ('A' for approval, 'P' for price)
        processing_type = "승인관리"
        if "구분" in df.columns and df["구분"].iloc[0].upper() == "P":
            processing_type = "가격관리"
            
        # Get current date for filename
        current_date = datetime.now().strftime("%Y%m%d")
        
        # Get base directory and filename
        base_dir = os.path.dirname(file_path)
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        
        # Calculate number of files needed
        total_rows = len(df)
        num_files = (total_rows + threshold - 1) // threshold  # Ceiling division
        
        split_files = []
        for i in range(num_files):
            start_idx = i * threshold
            end_idx = min((i + 1) * threshold, total_rows)
            
            # Create filename as per manual: 승인관리1(300)-날짜
            file_count = i + 1
            file_size = end_idx - start_idx
            split_filename = f"{processing_type}{file_count}({file_size})-{current_date}{ext}"
            split_path = os.path.join(base_dir, split_filename)
            
            # Save the chunk
            df.iloc[start_idx:end_idx].to_excel(split_path, index=False)
            split_files.append(split_path)
            
            logger.info(f"Created split file: {split_path} with {file_size} rows")
            
        return split_files
        
    except Exception as e:
        logger.error(f"Error splitting file: {e}")
        return [file_path]  # Return original file path if error occurs

def merge_result_files(file_paths: List[str], original_input: str) -> Optional[str]:
    """
    Merge multiple result files into a single file according to the manual requirements.
    
    Args:
        file_paths: List of paths to the result files
        original_input: Path to the original input file
        
    Returns:
        Path to the merged file or None if error
    """
    try:
        logger.info(f"Merging {len(file_paths)} result files")
        
        # Verify files exist
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.error(f"Result file not found: {file_path}")
                return None
        
        # Read all files into DataFrames
        dfs = []
        for file_path in file_paths:
            df = pd.read_excel(file_path)
            dfs.append(df)
        
        # Concatenate DataFrames
        if len(dfs) == 0:
            logger.error("No data to merge")
            return None
            
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Generate output filename
        base_dir = os.path.dirname(original_input)
        base_name, ext = os.path.splitext(os.path.basename(original_input))
        merged_filename = os.path.join(base_dir, f"{base_name}-merged-result{ext}")
        
        # Save merged file
        merged_df.to_excel(merged_filename, index=False)
        
        # Apply formatting
        try:
            wb = load_workbook(merged_filename)
            ws = wb.active
            
            # Format header
            header_font = ws.cell(row=1, column=1).font
            header_font.bold = True
            for col in range(1, ws.max_column + 1):
                ws.cell(row=1, column=col).font = header_font
            
            # Apply yellow highlighting for price differences
            yellow_fill_style = "FFFF00"
            for row in range(2, ws.max_row + 1):
                # Find price difference columns
                for col_name in ['가격차이(2)', '가격차이(3)']:
                    for col in range(1, ws.max_column + 1):
                        if ws.cell(row=1, column=col).value == col_name:
                            cell = ws.cell(row=row, column=col)
                            if isinstance(cell.value, (int, float)) and cell.value < 0:
                                from openpyxl.styles import PatternFill
                                cell.fill = PatternFill(start_color=yellow_fill_style, 
                                                       end_color=yellow_fill_style,
                                                       fill_type="solid")
            
            wb.save(merged_filename)
            
        except Exception as e:
            logger.error(f"Error applying formatting to merged file: {e}")
            # Continue with unformatted file
        
        logger.info(f"Created merged file: {merged_filename} with {len(merged_df)} rows")
        return merged_filename
        
    except Exception as e:
        logger.error(f"Error merging result files: {e}")
        return None 