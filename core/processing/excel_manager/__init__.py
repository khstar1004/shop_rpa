from .reader import ExcelReader
from .formatter import ExcelFormatter
from .writer import ExcelWriter
from .converter import ExcelConverter
from .postprocessor import ExcelPostProcessor
import pandas as pd
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

class ExcelManager:
    """Excel 파일 읽기, 쓰기, 포맷팅 관련 기능을 담당하는 클래스"""

    def __init__(self, config: dict, logger=None):
        """
        엑셀 매니저 초기화

        Args:
            config: 애플리케이션 설정
            logger: 로깅 인스턴스
        """
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
        
        # Extract Excel settings from config or use defaults
        self.excel_settings = processed_config.get("EXCEL", {})
        self._ensure_default_excel_settings()
            
        self.reader = ExcelReader(processed_config, logger)
        self.formatter = ExcelFormatter(processed_config, logger)
        self.writer = ExcelWriter(processed_config, logger)
        self.converter = ExcelConverter(processed_config, logger)
        self.postprocessor = ExcelPostProcessor(processed_config, logger)

    def _ensure_default_excel_settings(self):
        """설정에 필요한 기본값을 설정합니다."""
        defaults = {
            "sheet_name": "Sheet1",
            "start_row": 2,
            "required_columns": [
                "상품명",
                "판매단가(V포함)",
                "상품Code",
                "본사 이미지",
                "본사상품링크",
            ],
            "optional_columns": ["본사 단가", "가격", "상품코드"],
            "max_rows": 10000,
            "enable_formatting": True,
            "date_format": "YYYY-MM-DD",
            "number_format": "#,##0.00",
            "max_file_size_mb": 100,
            "enable_data_quality": True,
            "enable_duplicate_detection": True,
            "enable_auto_correction": True,
            "auto_correction_rules": ["price", "url", "product_code"],
            "report_formatting": True,
            "report_styles": True,
            "report_filters": True,
            "report_sorting": True,
            "report_freeze_panes": True,
            "report_auto_fit": True,
            "validation_rules": {
                "price": {"min": 0, "max": 1000000000},
                "product_code": {"pattern": r"^[A-Za-z0-9-]+$"},
                "url": {"pattern": r"^https?://.*$"},
            },
        }

        # Ensure top-level keys exist
        for key, value in defaults.items():
            if key not in self.excel_settings:
                self.excel_settings[key] = value
            # Ensure nested keys exist (specifically for validation_rules)
            elif key == "validation_rules" and isinstance(value, dict):
                if not isinstance(self.excel_settings[key], dict):
                    self.excel_settings[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key not in self.excel_settings[key]:
                        self.excel_settings[key][sub_key] = sub_value
                    elif isinstance(sub_value, dict):
                        if not isinstance(self.excel_settings[key][sub_key], dict):
                            self.excel_settings[key][sub_key] = {}
                        for item_key, item_value in sub_value.items():
                            if item_key not in self.excel_settings[key][sub_key]:
                                self.excel_settings[key][sub_key][item_key] = item_value

    def read_excel(self, file_path: str) -> pd.DataFrame:
        """Excel 파일을 읽고 DataFrame 반환"""
        return self.reader.read_excel_file(file_path)

    def read_excel_file(self, file_path: str) -> pd.DataFrame:
        """Excel 파일을 읽고 검증하여 DataFrame을 반환합니다. (이전 버전과의 호환성 유지)"""
        return self.reader.read_excel_file(file_path)

    def format_excel(self, file_path: str) -> None:
        """Excel 파일에 서식 적용"""
        self.formatter.apply_formatting_to_excel(file_path)

    def apply_formatting_to_excel(self, file_path: str) -> None:
        """Excel 파일에 서식 적용 (이전 버전과의 호환성 유지)"""
        self.formatter.apply_formatting_to_excel(file_path)

    def write_excel(self, products: list, output_path: str, sheet_name: str = None, naver_results: list = None) -> str:
        """제품 목록을 Excel 파일로 저장"""
        return self.writer.save_products(products, output_path, sheet_name, naver_results)

    def generate_enhanced_output(self, results: list, input_file: str, output_dir: str = None) -> str:
        """처리 결과를 엑셀로 저장하고 포맷팅을 적용합니다."""
        return self.writer.generate_enhanced_output(results, input_file, output_dir)

    def save_products(self, products: list, output_path: str, sheet_name: str = None, naver_results: list = None) -> str:
        """제품 목록을 Excel 파일로 저장 (이전 버전과의 호환성 유지)"""
        return self.writer.save_products(products, output_path, sheet_name, naver_results)

    def convert_xls_to_xlsx(self, input_directory: str) -> str:
        """XLS 파일을 XLSX 형식으로 변환"""
        return self.converter.convert_xls_to_xlsx(input_directory)

    def post_process_excel(self, file_path: str) -> str:
        """Excel 파일 후처리"""
        return self.postprocessor.post_process_excel_file(file_path)

    def post_process_excel_file(self, file_path: str) -> str:
        """Excel 파일 후처리 (이전 버전과의 호환성 유지)"""
        return self.postprocessor.post_process_excel_file(file_path)

    def add_hyperlinks(self, file_path: str) -> str:
        """Excel 파일에 하이퍼링크 추가"""
        return self.formatter.add_hyperlinks_to_excel(file_path)

    def add_hyperlinks_to_excel(self, file_path: str) -> str:
        """Excel 파일에 하이퍼링크 추가 (이전 버전과의 호환성 유지)"""
        return self.formatter.add_hyperlinks_to_excel(file_path)

    def filter_by_price_diff(self, file_path: str) -> str:
        """가격 차이가 있는 항목만 필터링"""
        return self.formatter.filter_excel_by_price_diff(file_path)

    def filter_excel_by_price_diff(self, file_path: str) -> str:
        """가격 차이가 있는 항목만 필터링 (이전 버전과의 호환성 유지)"""
        return self.formatter.filter_excel_by_price_diff(file_path)

    def remove_at_symbol(self, file_path: str) -> str:
        """@ 기호 제거 (여러 번 시도)"""
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

    def check_excel_file(self, file_path: str) -> None:
        """엑셀 파일의 필요한 컬럼이 있는지 확인하고 없으면 추가합니다."""
        try:
            self.logger.info(f"Checking Excel file: {file_path}")
            
            # 파일 존재 확인
            if not os.path.exists(file_path):
                self.logger.error(f"Excel file not found: {file_path}")
                return
            
            # 엑셀 파일 읽기
            df = pd.read_excel(file_path)
            
            # 추가할 컬럼 정의
            columns_to_add = ["본사 이미지", "고려기프트 이미지", "네이버 이미지"]
            need_to_modify = False
            
            # 컬럼 확인 및 추가
            for column in columns_to_add:
                if column not in df.columns:
                    df[column] = ""
                    need_to_modify = True
                    self.logger.info(f"Added missing column: {column}")
            
            # 수정 필요 없으면 종료
            if not need_to_modify:
                self.logger.info("All required columns exist. No modifications needed.")
                return
            
            # 컬럼명의 앞뒤 공백 제거
            df.columns = [col.strip() for col in df.columns]
            
            # 엑셀 파일 저장
            df.to_excel(file_path, index=False)
            self.logger.info(f"Updated Excel file with required columns: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error checking Excel file: {str(e)}", exc_info=True) 