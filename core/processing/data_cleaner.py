import pandas as pd
import re
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime

class DataCleaner:
    """데이터 정제 관련 기능을 담당하는 클래스"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        데이터 정제기 초기화
        
        Args:
            config: 애플리케이션 설정
            logger: 로깅 인스턴스
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 설정에서 검증 규칙 추출
        self.excel_settings = config.get('EXCEL', {})
        self._ensure_validation_rules()
    
    def _ensure_validation_rules(self):
        """기본 검증 규칙을 설정합니다."""
        default_rules = {
            'enable_auto_correction': True,
            'auto_correction_rules': ['price', 'url', 'product_code'],
            'validation_rules': {
                'price': {'min': 0, 'max': 1000000000},
                'product_code': {'pattern': r'^[A-Za-z0-9-]+$'},
                'url': {'pattern': r'^https?://.*$'}
            }
        }
        
        for key, value in default_rules.items():
            if key not in self.excel_settings:
                self.excel_settings[key] = value
            elif key == 'validation_rules' and isinstance(value, dict):
                if not isinstance(self.excel_settings[key], dict):
                    self.excel_settings[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key not in self.excel_settings[key]:
                        self.excel_settings[key][sub_key] = sub_value
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임 전체 정제 처리"""
        try:
            self.logger.info("Starting DataFrame cleaning process")
            df_copy = df.copy()
            
            # 중복 데이터 제거 (설정에 따라)
            if self.excel_settings.get('enable_duplicate_detection', True):
                duplicate_subset = ['상품명'] 
                if duplicate_subset[0] in df_copy.columns:
                    initial_count = len(df_copy)
                    df_copy = df_copy.drop_duplicates(subset=duplicate_subset, keep='first')
                    removed = initial_count - len(df_copy)
                    if removed > 0:
                        self.logger.info(f"Removed {removed} duplicate rows")
            
            # 컬럼별 정제 적용
            auto_correct = self.excel_settings.get('enable_auto_correction', True)
            rules = self.excel_settings.get('auto_correction_rules', [])
            
            # 상품명 정제
            if '상품명' in df_copy.columns:
                df_copy['상품명'] = self.clean_product_names(df_copy['상품명'])
            
            # 상품코드 정제
            if '상품Code' in df_copy.columns:
                for fallback in ['상품코드', 'Code', '업체코드']:
                    if fallback in df_copy.columns:
                        df_copy['상품Code'] = df_copy['상품Code'].fillna(df_copy[fallback])
                
                df_copy['상품Code'] = df_copy['상품Code'].apply(
                    lambda x: self.clean_product_code(x, auto_correct and 'product_code' in rules)
                )
            
            # 가격 정제
            price_col = '판매단가(V포함)'
            if price_col in df_copy.columns:
                min_price = self.excel_settings.get('validation_rules', {}).get('price', {}).get('min', 0)
                max_price = self.excel_settings.get('validation_rules', {}).get('price', {}).get('max', 1000000000)
                
                df_copy[price_col] = df_copy[price_col].apply(
                    lambda x: self.clean_price(x, min_price, max_price, auto_correct and 'price' in rules)
                )
            
            # URL 정제
            for url_col in ['본사 이미지', '본사상품링크']:
                if url_col in df_copy.columns:
                    df_copy[url_col] = df_copy[url_col].apply(
                        lambda x: self.clean_url(x, auto_correct and 'url' in rules)
                    )
            
            self.logger.info("DataFrame cleaning completed")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error cleaning DataFrame: {str(e)}", exc_info=True)
            return df  # 오류 발생 시 원본 반환
    
    def clean_product_names(self, names: Union[pd.Series, str]) -> Union[pd.Series, str]:
        """상품명 정제"""
        try:
            if isinstance(names, pd.Series):
                # 판다스 시리즈 처리
                cleaned = names.astype(str)
                # 앞에 붙은 숫자-패턴 제거
                cleaned = cleaned.str.replace(r'^\d+-', '', regex=True)
                # 특수문자 제거
                cleaned = cleaned.str.replace(r'[/()]', '', regex=True)
                # 괄호 안 숫자 제거
                cleaned = cleaned.str.replace(r'\(\d+\)', '', regex=True)
                # 공백 제거
                cleaned = cleaned.str.strip()
                return cleaned
            elif isinstance(names, str):
                # 단일 문자열 처리
                cleaned = re.sub(r'^\d+-', '', names)
                cleaned = re.sub(r'[/()]', '', cleaned)
                cleaned = re.sub(r'\(\d+\)', '', cleaned)
                return cleaned.strip()
            else:
                return str(names) if names is not None else ""
        except Exception as e:
            self.logger.error(f"Error cleaning product name: {str(e)}", exc_info=True)
            return names
    
    def clean_product_code(self, code: Any, auto_correct: bool) -> str:
        """상품 코드 정제"""
        try:
            if pd.isna(code) or not code:
                return ""
            
            code_str = str(code).strip()
            pattern = self.excel_settings.get('validation_rules', {}).get('product_code', {}).get('pattern', r'^[A-Za-z0-9\-]+$')
            
            # 유효성 검사
            is_valid = bool(re.match(pattern, code_str))
            if is_valid:
                return code_str
            
            # 자동 수정이 비활성화된 경우
            if not auto_correct:
                self.logger.debug(f"Invalid product code: {code_str} (auto-correction disabled)")
                return ""
            
            # 영문자, 숫자, 하이픈만 남기고 제거
            cleaned_code = re.sub(r'[^A-Za-z0-9\-]', '', code_str)
            
            if cleaned_code and re.match(pattern, cleaned_code):
                self.logger.debug(f"Auto-corrected product code: {code_str} -> {cleaned_code}")
                return cleaned_code
            else:
                # 유효한 코드 생성
                timestamp = int(datetime.now().timestamp())
                fallback_code = f"CODE-{timestamp}"
                self.logger.warning(f"Could not correct product code: {code_str}, using fallback: {fallback_code}")
                return fallback_code
                
        except Exception as e:
            self.logger.error(f"Error cleaning product code: {str(e)}", exc_info=True)
            return f"ERROR-{datetime.now().strftime('%H%M%S')}"
    
    def clean_price(self, price: Any, min_price: float, max_price: float, auto_correct: bool) -> float:
        """가격 정제"""
        try:
            if pd.isna(price):
                return 0.0
            
            # 문자열을 숫자로 변환
            if isinstance(price, str):
                # 숫자와 소수점만 남기기
                price_str = re.sub(r'[^\d.]', '', price)
                try:
                    price_float = float(price_str) if price_str else 0.0
                except ValueError:
                    self.logger.warning(f"Invalid price format: {price}")
                    return 0.0
            else:
                # 숫자 형식이면 그대로 사용
                price_float = float(price)
            
            # 유효성 검사
            if min_price <= price_float <= max_price:
                return price_float
            
            # 자동 수정이 비활성화된 경우
            if not auto_correct:
                self.logger.warning(f"Price out of range: {price_float} (min={min_price}, max={max_price})")
                return 0.0
            
            # 범위 내로 조정
            if price_float < min_price:
                self.logger.debug(f"Price too low: {price_float} -> {min_price}")
                return min_price
            else:  # price_float > max_price
                self.logger.debug(f"Price too high: {price_float} -> {max_price}")
                return max_price
                
        except Exception as e:
            self.logger.error(f"Error cleaning price: {str(e)}", exc_info=True)
            return 0.0
    
    def clean_url(self, url: Any, auto_correct: bool) -> str:
        """URL 정제"""
        try:
            if pd.isna(url) or not url or str(url).lower() == 'nan':
                return ""
            
            url_str = str(url).strip()
            pattern = self.excel_settings.get('validation_rules', {}).get('url', {}).get('pattern', r'^https?://.*$')
            
            # http:// 또는 https:// 로 시작하는지 확인
            if auto_correct and url_str and not url_str.startswith(('http://', 'https://')):
                url_str = 'https://' + url_str
                self.logger.debug(f"Added https:// prefix to URL: {url_str}")
            
            # 유효성 검사
            is_valid = bool(re.match(pattern, url_str))
            if is_valid:
                return url_str
            
            # 자동 수정이 비활성화되었거나 수정 후에도 유효하지 않은 경우
            self.logger.warning(f"Invalid URL format: {url_str}")
            return "" if auto_correct else url_str
            
        except Exception as e:
            self.logger.error(f"Error cleaning URL: {str(e)}", exc_info=True)
            return ""
    
    def validate_excel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """엑셀 데이터 유효성 검사"""
        try:
            self.logger.info("Validating DataFrame")
            validation_errors = []
            
            # 필수 컬럼 검사
            required_cols = self.excel_settings.get('required_columns', ['상품명', '판매단가(V포함)'])
            for col in required_cols:
                if col not in df.columns:
                    validation_errors.append(f"Required column '{col}' missing")
            
            # 빈 데이터 검사
            if '상품명' in df.columns:
                empty_names = df['상품명'].isna() | (df['상품명'].astype(str).str.strip() == '')
                if empty_names.any():
                    validation_errors.append(f"Found {empty_names.sum()} rows with empty product names")
            
            # 가격 범위 검사
            if '판매단가(V포함)' in df.columns:
                price_min = self.excel_settings.get('validation_rules', {}).get('price', {}).get('min', 0)
                price_max = self.excel_settings.get('validation_rules', {}).get('price', {}).get('max', 1000000000)
                
                invalid_prices = df[
                    (df['판매단가(V포함)'] < price_min) | 
                    (df['판매단가(V포함)'] > price_max)
                ]
                
                if not invalid_prices.empty:
                    validation_errors.append(f"Found {len(invalid_prices)} rows with invalid prices")
            
            # URL 형식 검사
            url_pattern = self.excel_settings.get('validation_rules', {}).get('url', {}).get('pattern', r'^https?://.*$')
            for url_col in ['본사 이미지', '본사상품링크']:
                if url_col in df.columns:
                    invalid_urls = df[~df[url_col].astype(str).str.match(url_pattern)]
                    if not invalid_urls.empty:
                        validation_errors.append(f"Found {len(invalid_urls)} rows with invalid URLs in '{url_col}'")
            
            # 오류 로깅
            if validation_errors:
                for error in validation_errors:
                    self.logger.warning(f"Validation error: {error}")
            else:
                self.logger.info("All validations passed")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating DataFrame: {str(e)}", exc_info=True)
            return df  # 오류 발생 시 원본 반환 