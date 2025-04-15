import os
import re
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any, Union


class ExcelConverter:
    def __init__(self, config: Union[Dict[str, Any], object], logger: Optional[logging.Logger] = None):
        """
        Excel 컨버터 초기화
        
        Args:
            config: 설정 객체
            logger: 로거 객체
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # config 객체의 타입 확인
        if hasattr(config, 'sections') and callable(getattr(config, 'get', None)):
            # ConfigParser 객체인 경우 설정값 가져오기
            self._get_config = lambda section, key, default=None: config.get(section, key, fallback=default)
        else:
            # dict 객체인 경우 설정값 가져오기
            self._get_config = lambda section, key, default=None: config.get(section, {}).get(key, default)

    def convert_xls_to_xlsx(self, input_directory: str) -> str:
        """
        XLS 파일을 XLSX 형식으로 변환합니다.

        Args:
            input_directory: 입력 디렉토리 경로

        Returns:
            str: 변환된 XLSX 파일 경로 또는 빈 문자열
        """
        try:
            self.logger.info(f"Looking for XLS files in: {input_directory}")

            # XLS 파일 찾기
            xls_files = [
                f for f in os.listdir(input_directory) if f.lower().endswith(".xls")
            ]

            if not xls_files:
                self.logger.warning("No XLS files found in the directory.")
                return ""

            # 첫 번째 XLS 파일 처리
            file_name = xls_files[0]
            file_path = os.path.join(input_directory, file_name)

            self.logger.info(f"Converting XLS file: {file_path}")

            # XLS 파일 로드
            # 여러 인코딩 시도
            tables = None
            encodings = ['cp949', 'utf-8', 'euc-kr']
            for encoding in encodings:
                try:
                    tables = pd.read_html(file_path, encoding=encoding)
                    self.logger.info(f"Successfully read file with encoding: {encoding}")
                    break
                except Exception as e:
                    self.logger.debug(f"Failed to read with encoding {encoding}: {str(e)}")
            
            if tables is None:
                # 마지막 시도: pandas ExcelFile 사용
                try:
                    df = pd.read_excel(file_path)
                    tables = [df]
                    self.logger.info("Successfully read file with pandas ExcelFile")
                except Exception as e:
                    self.logger.error(f"All attempts to read XLS file failed: {str(e)}")
                    return ""
            
            df = tables[0]

            # 첫 번째 행을 헤더로 사용 (필요한 경우)
            if any(col for col in df.columns if isinstance(col, int) or (isinstance(col, str) and col.isdigit())):
                self.logger.info("Using first row as header")
                df.columns = df.iloc[0].astype(str)
                df = df.drop(0)
            
            # 컬럼명이 None, NaN 또는 공백인 경우 처리
            df.columns = [f"Column_{i}" if pd.isna(col) or str(col).strip() == "" else str(col).strip() 
                         for i, col in enumerate(df.columns)]

            # 상품명 전처리 (// 이후 제거 및 특수 문자 정리)
            if "상품명" in df.columns:
                self.logger.info("Processing product names")
                df["상품명"] = df["상품명"].apply(self._preprocess_product_name)

            # 필요한 컬럼 추가
            for column in ["본사 이미지", "고려기프트 이미지", "네이버 이미지"]:
                if column not in df.columns:
                    df[column] = ""
                    self.logger.info(f"Added missing column: {column}")

            # 문자열에서 숫자로 변환 가능한 필드 처리
            numeric_columns = ["판매단가(V포함)", "가격", "단가", "Price"]
            for col in df.columns:
                for numeric_col in numeric_columns:
                    if numeric_col.lower() in col.lower():
                        try:
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                            df[col] = df[col].fillna(0)
                            self.logger.info(f"Converted column '{col}' to numeric")
                            break
                        except Exception as e:
                            self.logger.warning(f"Failed to convert column '{col}' to numeric: {str(e)}")

            # 출력 파일명 설정
            output_file_name = file_name.replace(".xls", ".xlsx")
            output_file_path = os.path.join(input_directory, output_file_name)

            # XLSX로 저장
            df.to_excel(output_file_path, index=False)
            self.logger.info(f"Converted file saved to: {output_file_path}")

            return output_file_path

        except Exception as e:
            self.logger.error(f"Error converting XLS to XLSX: {str(e)}", exc_info=True)
            return ""

    def _preprocess_product_name(self, product_name):
        """상품명 전처리 함수"""
        if not isinstance(product_name, str):
            return product_name

        try:
            # // 이후 제거
            if "//" in product_name:
                product_name = product_name.split("//")[0]

            # 특수 문자 및 패턴 제거
            pattern = r"(\d{4}_[A-Z]\.)|(\d+\+\d+)|[^a-zA-Z0-9가-힣\s]|\s+"
            product_name = re.sub(pattern, " ", product_name)
            
            # 불필요한 접두사/접미사 제거
            redundant_terms = [
                "정품", "NEW", "특가", "주문제작타올", "주문제작수건", 
                "결혼답례품 수건", "답례품수건", "주문제작 수건", 
                "돌답례품수건", "명절선물세트", "각종행사수건"
            ]
            
            for term in redundant_terms:
                product_name = product_name.replace(term, "")
                
            # 공백 정리
            product_name = product_name.strip()
            product_name = re.sub(" +", " ", product_name)
            
            return product_name
        except Exception as e:
            self.logger.warning(f"Error preprocessing product name: {str(e)}")
            return product_name

    def preprocess_product_name(self, product_name: str) -> str:
        """
        Preprocesses product names to make them more consistent and searchable.
        Removes special characters, normalizes whitespace, etc.
        """
        if not product_name or not isinstance(product_name, str):
            return ""
        
        # Remove common codes and special characters
        pattern = r"(\d{4}_[A-Z]\.)|(\d+\+\d+)|[^a-zA-Z0-9가-힣\s]|\s+"
        cleaned = re.sub(pattern, " ", product_name).strip()
        
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        return cleaned 