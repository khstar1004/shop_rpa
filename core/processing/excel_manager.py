import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from core.processing.excel_manager.reader import ExcelReader
from core.processing.excel_manager.formatter import ExcelFormatter
from core.processing.excel_manager.writer import ExcelWriter
from core.processing.excel_manager.converter import ExcelConverter
from core.processing.excel_manager.postprocessor import ExcelPostProcessor


class ExcelManager:
    """
    Excel Manager that integrates all Excel processing functionality.
    This is a facade that delegates to specialized subcomponents.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.excel_settings = config.get("EXCEL", {})
        
        # Initialize all components
        self.reader = ExcelReader(config, logger)
        self.formatter = ExcelFormatter(config, logger)
        self.writer = ExcelWriter(config, logger)
        self.converter = ExcelConverter(config, logger)
        self.postprocessor = ExcelPostProcessor(config, logger)
        
        # Make sure default settings are available
        self._ensure_default_excel_settings()
    
    # Delegate to component methods
    
    def _ensure_default_excel_settings(self):
        # This is already handled in ExcelReader, but we keep the method for backward compatibility
        self.reader._ensure_default_excel_settings()
    
    def _clean_url(self, url: str) -> str:
        return self.reader._clean_url(url)
    
    def _normalize_url(self, url: str) -> str:
        # For backward compatibility
        return self._clean_url(url)
    
    def _compute_price_metrics(self, base_price, compare_price):
        return self.reader._compute_price_metrics(base_price, compare_price)
    
    def read_excel_file(self, file_path: str) -> pd.DataFrame:
        return self.reader.read_excel_file(file_path)
    
    def _create_minimal_error_dataframe(self, error_message: str) -> pd.DataFrame:
        return self.reader._create_minimal_error_dataframe(error_message)
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.reader._ensure_required_columns(df)
    
    def _find_similar_column(self, df: pd.DataFrame, target_column: str):
        return self.reader._find_similar_column(df, target_column)
    
    def apply_formatting_to_excel(self, excel_file: str) -> None:
        return self.formatter.apply_formatting_to_excel(excel_file)
    
    def generate_enhanced_output(self, results: List, input_file: str, output_dir: Optional[str] = None) -> str:
        return self.writer.generate_enhanced_output(results, input_file, output_dir)
    
    def _calculate_price_differences(self, df: pd.DataFrame) -> None:
        """Calculate price differences between columns and add the results to the dataframe"""
        try:
            # 1. Calculate price difference for "고려기프트" prices
            if "판매단가(V포함)" in df.columns and "판매단가(V포함)(2)" in df.columns:
                # Remove existing columns if they exist
                if "가격차이(2)" in df.columns:
                    del df["가격차이(2)"]
                if "가격차이(2)(%)" in df.columns:
                    del df["가격차이(2)(%)"]
                
                # Calculate price differences (koryo price - base price)
                df["가격차이(2)"], df["가격차이(2)(%)"] = zip(*df.apply(
                    lambda row: self._compute_price_metrics(row.get("판매단가(V포함)"), row.get("판매단가(V포함)(2)")), 
                    axis=1
                ))
            
            # 2. Calculate price difference for "네이버" prices
            if "판매단가(V포함)" in df.columns and "판매단가(V포함)(3)" in df.columns:
                # Remove existing columns if they exist
                if "가격차이(3)" in df.columns:
                    del df["가격차이(3)"]
                if "가격차이(3)(%)" in df.columns:
                    del df["가격차이(3)(%)"]
                
                # Calculate price differences (naver price - base price)
                df["가격차이(3)"], df["가격차이(3)(%)"] = zip(*df.apply(
                    lambda row: self._compute_price_metrics(row.get("판매단가(V포함)"), row.get("판매단가(V포함)(3)")), 
                    axis=1
                ))
        except Exception as e:
            self.logger.error(f"Error calculating price differences: {str(e)}")
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns in a standardized way for output"""
        # Standard column order (matching example Excel files)
        column_order = [
            "구분", "담당자", "업체명", "업체코드", "Code", "상품Code", "중분류카테고리",
            "상품명", "기본수량(1)", "판매단가(V포함)", "본사상품링크",
            
            # Koryo Gift columns
            "기본수량(2)", "판매단가(V포함)(2)", "판매가(V포함)(2)",
            "가격차이(2)", "가격차이(2)(%)", "매칭_상황(2)", "텍스트유사도(2)",
            "고려기프트 상품링크", "고려기프트 이미지",
            
            # Naver columns
            "기본수량(3)", "판매단가(V포함)(3)", "가격차이(3)", "가격차이(3)(%)",
            "매칭_상황(3)", "텍스트유사도(3)", "공급사명", "네이버 쇼핑 링크",
            "공급사 상품링크", "네이버 이미지",
            
            # Base image
            "본사 이미지",
        ]
        
        # Filter to only include columns that exist in the dataframe
        existing_columns = [col for col in column_order if col in df.columns]
        
        # Add any columns that exist in the dataframe but aren't in our standard order
        missing_columns = [col for col in df.columns if col not in column_order]
        
        # Create the final ordered columns list
        ordered_columns = existing_columns + missing_columns
        
        # Return reordered dataframe
        return df[ordered_columns]
    
    def _ensure_required_fields(self, row: Dict, result) -> None:
        """Ensure all required fields exist in the row dictionary"""
        # Basic required fields
        required_fields = [
            "구분", "담당자", "업체명", "업체코드", "Code", "상품Code",
            "중분류카테고리", "상품명", "기본수량(1)", "판매단가(V포함)", "본사상품링크",
        ]
        
        # Make sure all required fields exist
        for field in required_fields:
            if field not in row:
                # Handle Code and 상품Code interchangeably
                if field == "Code" and "상품Code" in row:
                    row["Code"] = row["상품Code"]
                elif field == "상품Code" and "Code" in row:
                    row["상품Code"] = row["Code"]
                else:
                    row[field] = ""
        
        # Handle image URLs
        if "본사 이미지" not in row and hasattr(result.source_product, "image_url") and result.source_product.image_url:
            image_url = self._clean_url(result.source_product.image_url)
            if image_url.startswith('https:'):
                row["본사 이미지"] = f'=IMAGE("{image_url}", 2)'
            else:
                row["본사 이미지"] = image_url
        elif "본사 이미지" in row and isinstance(row["본사 이미지"], str) and row["본사 이미지"].startswith('http'):
            image_url = self._clean_url(row["본사 이미지"])
            if image_url.startswith('https:'):
                row["본사 이미지"] = f'=IMAGE("{image_url}", 2)'
            else:
                row["본사 이미지"] = image_url
                
        # Handle product URLs
        if "본사상품링크" not in row and hasattr(result.source_product, "url"):
            row["본사상품링크"] = self._clean_url(result.source_product.url)
        elif "본사상품링크" in row and isinstance(row["본사상품링크"], str):
            row["본사상품링크"] = self._clean_url(row["본사상품링크"])
    
    def _validate_critical_fields(self, row: Dict, result) -> None:
        """Validate and fix critical fields in the row dictionary"""
        # Ensure product name
        if not row.get("상품명") and hasattr(result.source_product, "name"):
            row["상품명"] = result.source_product.name
            
        # Ensure price
        if not row.get("판매단가(V포함)") and hasattr(result.source_product, "price"):
            row["판매단가(V포함)"] = result.source_product.price
            
        # Ensure product code
        if (not row.get("상품Code") and not row.get("Code") and 
            hasattr(result.source_product, "id")):
            row["상품Code"] = result.source_product.id
    
    def _add_koryo_match_data(self, row: Dict, koryo_match) -> None:
        """Add Koryo Gift matching data to the row dictionary"""
        try:
            if hasattr(koryo_match, "matched_product"):
                match_product = koryo_match.matched_product
                
                # Check if match is successful based on text similarity threshold
                similarity_threshold = self.config.get("MATCHING", {}).get("TEXT_SIMILARITY_THRESHOLD", 0.85)
                match_success = getattr(koryo_match, "text_similarity", 0) >= similarity_threshold
                
                # Add price information
                if hasattr(match_product, "price"):
                    row["판매단가(V포함)(2)"] = match_product.price
                
                # Add price difference information
                if hasattr(koryo_match, "price_difference"):
                    row["가격차이(2)"] = koryo_match.price_difference
                
                if hasattr(koryo_match, "price_difference_percent"):
                    row["가격차이(2)(%)"] = koryo_match.price_difference_percent
                
                # Add text similarity
                if hasattr(koryo_match, "text_similarity"):
                    row["텍스트유사도(2)"] = round(koryo_match.text_similarity, 2)
                
                # Handle image URL
                if hasattr(match_product, "image_url") and match_product.image_url:
                    image_url = self._clean_url(match_product.image_url)
                    if image_url.startswith('https:'):
                        row["고려기프트 이미지"] = f'=IMAGE("{image_url}", 2)'
                    else:
                        row["고려기프트 이미지"] = image_url
                else:
                    row["고려기프트 이미지"] = "이미지를 찾을 수 없음"
                
                # Handle product URL
                if hasattr(match_product, "url") and match_product.url:
                    row["고려기프트 상품링크"] = self._clean_url(match_product.url)
                else:
                    row["고려기프트 상품링크"] = "상품 링크를 찾을 수 없음"
                
                # Set match status
                if not match_success:
                    # If match isn't successful, compile reasons why
                    text_similarity = getattr(koryo_match, "text_similarity", 0)
                    category_match = getattr(koryo_match, "category_match", True)
                    brand_match = getattr(koryo_match, "brand_match", True)
                    price_in_range = getattr(koryo_match, "price_in_range", True)
                    
                    error_messages = []
                    if text_similarity < similarity_threshold:
                        error_messages.append(f"유사도: {text_similarity:.2f}")
                    if not category_match:
                        error_messages.append("카테고리 불일치")
                    if not brand_match:
                        error_messages.append("브랜드 불일치")
                    if not price_in_range:
                        error_messages.append("가격 범위 초과")
                        
                    row["매칭_상황(2)"] = f"매칭 실패 ({', '.join(error_messages)})"
            else:
                # No matched product
                row["매칭_상황(2)"] = "고려기프트에서 상품을 찾지 못했습니다"
                row["판매단가(V포함)(2)"] = 0
                row["가격차이(2)"] = 0
                row["가격차이(2)(%)"] = 0
                row["고려기프트 이미지"] = "상품을 찾을 수 없음"
                row["고려기프트 상품링크"] = "상품을 찾을 수 없음"
        except Exception as e:
            self.logger.error(f"Error adding Koryo match data: {str(e)}")
            row["매칭_상황(2)"] = f"오류 발생: {str(e)}"
    
    def _add_naver_match_data(self, row: Dict, naver_match) -> None:
        """Add Naver matching data to the row dictionary"""
        try:
            if hasattr(naver_match, "matched_product"):
                match_product = naver_match.matched_product
                
                # Check if match is successful based on text similarity threshold
                similarity_threshold = self.config.get("MATCHING", {}).get("TEXT_SIMILARITY_THRESHOLD", 0.85)
                match_success = getattr(naver_match, "text_similarity", 0) >= similarity_threshold
                
                # Add supplier information
                if hasattr(match_product, "brand") and match_product.brand:
                    row["공급사명"] = match_product.brand
                else:
                    row["공급사명"] = "정보 없음"
                
                # Add price information
                if hasattr(match_product, "price"):
                    row["판매단가(V포함)(3)"] = match_product.price
                
                # Add price difference information
                if hasattr(naver_match, "price_difference"):
                    row["가격차이(3)"] = naver_match.price_difference
                
                if hasattr(naver_match, "price_difference_percent"):
                    row["가격차이(3)(%)"] = naver_match.price_difference_percent
                
                # Add text similarity
                if hasattr(naver_match, "text_similarity"):
                    row["텍스트유사도(3)"] = round(naver_match.text_similarity, 2)
                
                # Handle image URL
                if hasattr(match_product, "image_url") and match_product.image_url:
                    image_url = self._clean_url(match_product.image_url)
                    if image_url.startswith('https:'):
                        row["네이버 이미지"] = f'=IMAGE("{image_url}", 2)'
                    else:
                        row["네이버 이미지"] = image_url
                else:
                    row["네이버 이미지"] = "이미지를 찾을 수 없음"
                
                # Handle product URLs
                if hasattr(match_product, "url") and match_product.url:
                    row["네이버 쇼핑 링크"] = self._clean_url(match_product.url)
                    row["공급사 상품링크"] = self._clean_url(match_product.url)
                else:
                    row["네이버 쇼핑 링크"] = "상품 링크를 찾을 수 없음"
                    row["공급사 상품링크"] = "상품 링크를 찾을 수 없음"
                
                # Set match status
                if not match_success and not row.get("매칭_상황(3)"):
                    # If match isn't successful, compile reasons why
                    text_similarity = getattr(naver_match, "text_similarity", 0)
                    category_match = getattr(naver_match, "category_match", True)
                    brand_match = getattr(naver_match, "brand_match", True)
                    price_in_range = getattr(naver_match, "price_in_range", True)
                    
                    error_messages = []
                    if text_similarity < similarity_threshold:
                        error_messages.append(f"유사도: {text_similarity:.2f}")
                    if not category_match:
                        error_messages.append("카테고리 불일치")
                    if not brand_match:
                        error_messages.append("브랜드 불일치")
                    if not price_in_range:
                        error_messages.append("가격 범위 초과")
                        
                    row["매칭_상황(3)"] = f"매칭 실패 ({', '.join(error_messages)})"
            else:
                # No matched product
                row["매칭_상황(3)"] = "네이버에서 상품을 찾지 못했습니다"
                row["판매단가(V포함)(3)"] = 0
                row["가격차이(3)"] = 0
                row["가격차이(3)(%)"] = 0
                row["공급사명"] = "상품을 찾을 수 없음"
                row["네이버 이미지"] = "상품을 찾을 수 없음"
                row["네이버 쇼핑 링크"] = "상품을 찾을 수 없음"
                row["공급사 상품링크"] = "상품을 찾을 수 없음"
        except Exception as e:
            self.logger.error(f"Error adding Naver match data: {str(e)}")
            row["매칭_상황(3)"] = f"오류 발생: {str(e)}"
    
    def _is_price_in_range(self, price) -> bool:
        """Check if the price is within the valid range defined in settings"""
        try:
            price_value = float(price)
            min_price = float(
                self.excel_settings.get("validation_rules", {})
                .get("price", {})
                .get("min", 0)
            )
            max_price = float(
                self.excel_settings.get("validation_rules", {})
                .get("price", {})
                .get("max", 1000000000)
            )
            return min_price <= price_value <= max_price
        except (ValueError, TypeError):
            return False
    
    def _set_default_values(self, row: Dict) -> None:
        """Set default values for empty fields in the row dictionary"""
        default_values = {
            "상품명": "",
            "상품Code": "DEFAULT-CODE",
            "판매단가(V포함)": 0,
            "본사상품링크": "",
            "source": "haeoreum"
        }
        
        for field, default_value in default_values.items():
            if field not in row or pd.isna(row[field]) or row[field] == "":
                row[field] = default_value
    
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
    
    def convert_xls_to_xlsx(self, input_directory: str) -> str:
        return self.converter.convert_xls_to_xlsx(input_directory)
    
    def add_hyperlinks_to_excel(self, file_path: str) -> str:
        return self.formatter.add_hyperlinks_to_excel(file_path)
    
    def filter_excel_by_price_diff(self, file_path: str) -> str:
        return self.formatter.filter_excel_by_price_diff(file_path)
    
    def _apply_filter_rules(self, worksheet):
        """Apply filtering rules to the worksheet for price difference filtering"""
        from urllib.parse import urlparse
        
        # Process in reverse order to handle row deletion
        for row in reversed(list(worksheet.iter_rows(min_row=2, max_row=worksheet.max_row))):
            # Find column indices
            goleo_quantity_col = None
            naver_quantity_col = None
            goleo_link_col = None
            naver_link_col = None
            
            # Find column indices from headers
            for i, cell in enumerate(worksheet[1]):
                if cell.value == "고려 기본수량":
                    goleo_quantity_col = i
                elif cell.value == "네이버 기본수량":
                    naver_quantity_col = i
                elif cell.value == "고려 링크":
                    goleo_link_col = i
                elif cell.value == "네이버 링크":
                    naver_link_col = i
            
            # Check if row should be deleted due to "품절" status
            if (goleo_quantity_col is not None and 
                row[goleo_quantity_col].value == "품절"):
                worksheet.delete_rows(row[0].row)
                continue
                
            if (naver_quantity_col is not None and 
                row[naver_quantity_col].value == "품절"):
                worksheet.delete_rows(row[0].row)
                continue
            
            # Validate URLs
            if goleo_link_col is not None:
                link_value = row[goleo_link_col].value
                if (link_value and isinstance(link_value, str) and 
                    not bool(urlparse(link_value).scheme)):
                    row[goleo_link_col].value = None
                    
            if naver_link_col is not None:
                link_value = row[naver_link_col].value
                if (link_value and isinstance(link_value, str) and 
                    not bool(urlparse(link_value).scheme)):
                    row[naver_link_col].value = None
    
    def _write_product_data(self, worksheet, row_idx: int, product) -> int:
        return self.writer._write_product_data(worksheet, row_idx, product)
    
    def create_worksheet(self, workbook, sheet_name: str):
        return self.writer.create_worksheet(workbook, sheet_name)
    
    def save_products(self, products: List, output_path: str, sheet_name: str = None, naver_results: List = None):
        return self.writer.save_products(products, output_path, sheet_name, naver_results)
    
    def remove_at_symbol(self, file_path: str) -> str:
        return self.postprocessor.remove_at_symbol(file_path)
    
    def post_process_excel_file(self, file_path: str) -> str:
        return self.postprocessor.post_process_excel_file(file_path)
    
    def _fix_image_formulas(self, file_path: str) -> None:
        return self.postprocessor._fix_image_formulas(file_path)
    
    def save_excel_file(self, df: pd.DataFrame, file_path: str) -> None:
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"Created output directory: {output_dir}")
                
            df.to_excel(file_path, index=False)
            self.logger.info(f"DataFrame saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving Excel file: {str(e)}", exc_info=True)
            raise
