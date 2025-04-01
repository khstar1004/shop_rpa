import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, List, Tuple
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
import os
import re

class ExcelManager:
    """Excel 파일 읽기, 쓰기, 포맷팅 관련 기능을 담당하는 클래스"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        엑셀 매니저 초기화
        
        Args:
            config: 애플리케이션 설정
            logger: 로깅 인스턴스
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract Excel settings from config or use defaults
        self.excel_settings = config.get('EXCEL', {})
        self._ensure_default_excel_settings()
        
        # Yellow fill for price differences
        self.price_difference_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    
    def _ensure_default_excel_settings(self):
        """설정에 필요한 기본값을 설정합니다."""
        defaults = {
            'sheet_name': 'Sheet1',
            'start_row': 2,
            'required_columns': ['상품명', '판매단가(V포함)', '상품Code', '본사 이미지', '본사상품링크'],
            'optional_columns': ['본사 단가', '가격', '상품코드'],
            'max_rows': 10000,
            'enable_formatting': True,
            'date_format': 'YYYY-MM-DD',
            'number_format': '#,##0.00',
            'max_file_size_mb': 100,
            'enable_data_quality': True,
            'enable_duplicate_detection': True,
            'enable_auto_correction': True,
            'auto_correction_rules': ['price', 'url', 'product_code'],
            'report_formatting': True,
            'report_styles': True,
            'report_filters': True,
            'report_sorting': True,
            'report_freeze_panes': True,
            'report_auto_fit': True,
            'validation_rules': {
                'price': {'min': 0, 'max': 1000000000},
                'product_code': {'pattern': r'^[A-Za-z0-9-]+$'},
                'url': {'pattern': r'^https?://.*$'}
            }
        }
        
        # Ensure top-level keys exist
        for key, value in defaults.items():
            if key not in self.excel_settings:
                self.excel_settings[key] = value
            # Ensure nested keys exist (specifically for validation_rules)
            elif key == 'validation_rules' and isinstance(value, dict):
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
    
    def read_excel_file(self, file_path: str) -> pd.DataFrame:
        """Excel 파일을 읽고 검증하여 DataFrame을 반환합니다."""
        try:
            # 파일 존재 확인
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.error(f"Excel file not found: {file_path}")
                return self._create_minimal_error_dataframe("File not found")
            
            self.logger.info(f"Reading Excel file: {file_path}")
            
            # 시트 확인
            sheet_name = self.excel_settings['sheet_name']
            sheet_names = []
            
            try:
                workbook = load_workbook(file_path, read_only=True, data_only=True)
                sheet_names = workbook.sheetnames
                workbook.close()  # Close workbook to free resources
                
                if sheet_name not in sheet_names and sheet_names:
                    self.logger.warning(f"Sheet '{sheet_name}' not found. Using first sheet: '{sheet_names[0]}'")
                    sheet_name = sheet_names[0]
            except Exception as e:
                self.logger.warning(f"Could not inspect Excel structure: {str(e)}. Using default sheet name.")
            
            # 다양한 방법으로 읽기 시도
            all_dataframes = []
            
            # 모든 시트 읽기 시도
            try:
                all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
                for sheet, df in all_sheets.items():
                    if not df.empty:
                        self.logger.info(f"Found data in sheet '{sheet}' with {len(df)} rows")
                        all_dataframes.append((df, sheet, 0))
            except Exception as e:
                self.logger.warning(f"Failed to read all sheets: {str(e)}")
            
            # 첫 행 스킵 시도 (헤더가 있는 경우)
            if not all_dataframes:
                for skip_rows in range(6):  # 최대 5개 행 스킵 시도
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows, engine='openpyxl')
                        if not df.empty:
                            self.logger.info(f"Found data with skiprows={skip_rows}")
                            all_dataframes.append((df, sheet_name, skip_rows))
                    except Exception as e:
                        self.logger.debug(f"Failed with skiprows={skip_rows}: {str(e)}")
            
            # 대체 엔진 시도
            if not all_dataframes:
                try:
                    # xlrd 엔진 시도 (오래된 Excel 형식)
                    df = pd.read_excel(file_path, engine='xlrd')
                    if not df.empty:
                        all_dataframes.append((df, "Unknown", 0))
                except Exception:
                    pass
                
                try:
                    # CSV 형식 시도
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        all_dataframes.append((df, "CSV", 0))
                except Exception:
                    pass
            
            # 데이터프레임이 없으면 기본 형식 반환
            if not all_dataframes:
                self.logger.error(f"Could not read any valid data from {file_path}")
                return self._create_minimal_error_dataframe("No data found")
            
            # 가장 많은 행을 가진 데이터프레임 선택
            best_df, sheet, skiprows = max(all_dataframes, key=lambda x: len(x[0]))
            self.logger.info(f"Selected dataframe from sheet '{sheet}' with {len(best_df)} rows")
            
            # 필수 컬럼 확인 및 추가
            best_df = self._ensure_required_columns(best_df)
            
            return best_df
            
        except Exception as e:
            self.logger.error(f"Error reading Excel file: {str(e)}", exc_info=True)
            return self._create_minimal_error_dataframe(f"Error: {str(e)}")
    
    def _create_minimal_error_dataframe(self, error_message: str) -> pd.DataFrame:
        """오류 상황에서 기본 데이터프레임을 생성합니다."""
        required_columns = self.excel_settings['required_columns']
        error_data = {col: [""] for col in required_columns}
        
        # 첫 번째 행에 오류 메시지 표시
        if '상품명' in required_columns:
            error_data['상품명'] = [f"Error: {error_message}"]
        
        if '판매단가(V포함)' in required_columns:
            error_data['판매단가(V포함)'] = [0]
            
        if '상품Code' in required_columns:
            error_data['상품Code'] = ["ERROR"]
            
        return pd.DataFrame(error_data)
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임에 필수 컬럼이 있는지 확인하고 없으면 추가합니다."""
        df_copy = df.copy()
        required_columns = self.excel_settings['required_columns']
        
        for col in required_columns:
            if col not in df_copy.columns:
                # 유사한 컬럼 찾기
                similar_col = self._find_similar_column(df_copy, col)
                
                if similar_col:
                    self.logger.info(f"Mapped '{similar_col}' to required column '{col}'")
                    df_copy[col] = df_copy[similar_col]
                else:
                    self.logger.warning(f"Creating default values for missing column '{col}'")
                    # 컬럼 유형에 따른 기본값 설정
                    if '단가' in col or '가격' in col:
                        df_copy[col] = 0
                    elif 'Code' in col or '코드' in col:
                        df_copy[col] = [f"CODE-{i+1}" for i in range(len(df_copy))]
                    elif '이미지' in col or '링크' in col:
                        df_copy[col] = ""
                    else:
                        df_copy[col] = [f"Item {i+1}" for i in range(len(df_copy))]
        
        return df_copy
    
    def _find_similar_column(self, df: pd.DataFrame, target_column: str) -> Optional[str]:
        """데이터프레임에서 타겟 컬럼과 유사한 컬럼을 찾습니다."""
        # 한국어 컬럼명 매핑
        column_mapping = {
            '상품명': ['품명', '제품명', '상품', 'product', 'name', 'item', '품목', '상품이름'],
            '판매단가(V포함)': ['단가', '판매가', '가격', 'price', '가격(v포함)', '단가(vat)', '판매단가'],
            '상품Code': ['코드', 'code', 'item code', 'product code', '품목코드', '제품코드', '상품코드'],
            '본사 이미지': ['이미지', 'image', '상품이미지', '제품이미지', '이미지주소', 'image url'],
            '본사상품링크': ['링크', 'link', 'url', '상품링크', '제품링크', '상품url', '제품url', '홈페이지']
        }
        
        # 대소문자 무시, 유사성 검사
        for col in df.columns:
            if col.lower() == target_column.lower():
                return col
        
        # 매핑 검사
        if target_column in column_mapping:
            for similar in column_mapping[target_column]:
                for col in df.columns:
                    if similar.lower() in col.lower():
                        return col
        
        return None
    
    def apply_formatting_to_excel(self, excel_file: str) -> None:
        """엑셀 파일에 포맷팅을 적용합니다."""
        try:
            wb = load_workbook(excel_file)
            ws = wb.active
            
            # 가격차이 컬럼 찾기
            price_diff_cols = []
            for col in range(1, ws.max_column + 1):
                header = ws.cell(row=1, column=col).value
                if header and ('가격차이' in str(header)):
                    price_diff_cols.append(col)
            
            # 음수 가격차이에 노란색 하이라이트 적용
            for col in price_diff_cols:
                for row in range(2, ws.max_row + 1):
                    cell = ws.cell(row=row, column=col)
                    try:
                        value = float(cell.value) if cell.value is not None else 0
                        if value < 0:
                            cell.fill = self.price_difference_fill
                    except:
                        pass  # 숫자가 아닌 셀은 무시
            
            # 변경사항 저장
            wb.save(excel_file)
            self.logger.info(f"Formatting applied to {excel_file}")
            
        except Exception as e:
            self.logger.error(f"Error applying formatting: {str(e)}", exc_info=True)
    
    def generate_enhanced_output(self, results: List, input_file: str) -> str:
        """처리 결과를 엑셀로 저장하고 포맷팅을 적용합니다."""
        try:
            # 결과 데이터 준비
            report_data = []
            for result in results:
                if hasattr(result, 'source_product') and result.source_product:
                    row = {}
                    # 원본 제품 데이터 추출
                    source_data = result.source_product.original_input_data
                    for key, value in source_data.items():
                        row[key] = value
                    
                    # 고려기프트 매칭 데이터 추가
                    if hasattr(result, 'best_koryo_match') and result.best_koryo_match:
                        row['판매단가(V포함)(2)'] = result.best_koryo_match.matched_product.price
                        row['가격차이(2)'] = result.best_koryo_match.price_difference
                        row['가격차이(2)%'] = result.best_koryo_match.price_difference_percent
                        row['고려기프트 이미지'] = result.best_koryo_match.matched_product.image_url
                        row['고려기프트상품링크'] = result.best_koryo_match.matched_product.url
                    
                    # 네이버 매칭 데이터 추가
                    if hasattr(result, 'best_naver_match') and result.best_naver_match:
                        row['판매단가(V포함)(3)'] = result.best_naver_match.matched_product.price
                        row['가격차이(3)'] = result.best_naver_match.price_difference
                        row['가격차이(3)%'] = result.best_naver_match.price_difference_percent
                        row['네이버 이미지'] = result.best_naver_match.matched_product.image_url
                        row['네이버쇼핑 링크'] = result.best_naver_match.matched_product.url
                        if hasattr(result.best_naver_match.matched_product, 'brand'):
                            row['공급사명'] = result.best_naver_match.matched_product.brand
                    
                    report_data.append(row)
            
            # DataFrame 생성
            result_df = pd.DataFrame(report_data)
            
            # 파일명 생성
            output_file = f"{os.path.splitext(input_file)[0]}-result.xlsx"
            
            # 엑셀로 저장
            result_df.to_excel(output_file, index=False)
            
            # 포맷팅 적용
            self.apply_formatting_to_excel(output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating output: {str(e)}", exc_info=True)
            return "" 