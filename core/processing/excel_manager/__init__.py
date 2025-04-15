from .reader import ExcelReader
from .formatter import ExcelFormatter
from .writer import ExcelWriter
from .converter import ExcelConverter
from .postprocessor import ExcelPostProcessor
import pandas as pd
import os
import time
import logging

class ExcelManager:
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 수정된 부분: config 객체가 ConfigParser인지 dict인지 확인
        # reader, formatter 등 생성 전에 config를 적절히 처리
        if hasattr(config, 'sections') and callable(config.get):
            # ConfigParser 객체인 경우 dict로 변환하여 전달
            config_dict = {}
            for section in config.sections():
                config_dict[section] = dict(config[section])
            processed_config = config_dict
        else:
            # 이미 dict인 경우 그대로 사용
            processed_config = config
            
        self.reader = ExcelReader(processed_config, logger)
        self.formatter = ExcelFormatter(processed_config, logger)
        self.writer = ExcelWriter(processed_config, logger)
        self.converter = ExcelConverter(processed_config, logger)
        self.postprocessor = ExcelPostProcessor(processed_config, logger)

    def read_excel(self, file_path: str) -> pd.DataFrame:
        return self.reader.read_excel_file(file_path)

    def format_excel(self, file_path: str) -> None:
        self.formatter.apply_formatting_to_excel(file_path)

    def write_excel(self, products: list, output_path: str, sheet_name: str = None, naver_results: list = None) -> str:
        return self.writer.save_products(products, output_path, sheet_name, naver_results)

    def convert_xls_to_xlsx(self, input_directory: str) -> str:
        return self.converter.convert_xls_to_xlsx(input_directory)

    def post_process_excel(self, file_path: str) -> str:
        return self.postprocessor.post_process_excel_file(file_path)

    def add_hyperlinks(self, file_path: str) -> str:
        return self.formatter.add_hyperlinks_to_excel(file_path)

    def filter_by_price_diff(self, file_path: str) -> str:
        return self.formatter.filter_excel_by_price_diff(file_path)

    def remove_at_symbol(self, file_path: str) -> str:
        # Try to remove @ symbols with retries
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                return self.postprocessor.remove_at_symbol(file_path)
            except PermissionError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to remove @ symbols after {max_retries} attempts: {str(e)}")
                    raise

    def save_products(self, products: list, output_path: str, sheet_name: str = None, naver_results: list = None) -> str:
        return self.writer.save_products(products, output_path, sheet_name, naver_results)

    def check_excel_file(self, file_path: str) -> None:
        """Check if Excel file has the required columns and add them if missing"""
        try:
            self.logger.info(f"Checking Excel file: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"Excel file not found: {file_path}")
                return
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Define columns to add if missing
            columns_to_add = ["본사 이미지", "고려기프트 이미지", "네이버 이미지"]
            need_to_modify = False
            
            # Check and add missing columns
            for column in columns_to_add:
                if column not in df.columns:
                    df[column] = ""
                    need_to_modify = True
                    self.logger.info(f"Added missing column: {column}")
            
            # If no modifications needed, return
            if not need_to_modify:
                self.logger.info("All required columns exist. No modifications needed.")
                return
            
            # Clean column names (remove whitespace)
            df.columns = [col.strip() for col in df.columns]
            
            # Save Excel file
            df.to_excel(file_path, index=False)
            self.logger.info(f"Updated Excel file with required columns: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error checking Excel file: {str(e)}")

    def post_process_excel_file(self, file_path: str) -> str:
        """Alias for post_process_excel to maintain backward compatibility"""
        return self.post_process_excel(file_path) 