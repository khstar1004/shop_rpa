import logging
import os
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
from openpyxl import load_workbook


class ExcelReader:
    def __init__(self, config: dict, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.excel_settings = config.get("EXCEL", {})
        self._ensure_default_excel_settings()

    def _ensure_default_excel_settings(self) -> None:
        defaults = {
            "sheet_name": "Sheet1",
            "start_row": 2,
            "required_columns": [
                "상품명",
                "판매단가(V포함)",
                "상품Code",
                "본사 이미지",
                "본사상품링크"
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
                "url": {"pattern": r"^https?://.*$"}
            }
        }
        for key, value in defaults.items():
            if key not in self.excel_settings:
                self.excel_settings[key] = value
            elif key == "validation_rules" and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in self.excel_settings[key]:
                        self.excel_settings[key][sub_key] = sub_value

    def _clean_url(self, url: str) -> str:
        if not url or not isinstance(url, str):
            return ""
        cleaned_url = url.strip().replace('@', '')
        if cleaned_url.startswith("http:"):
            cleaned_url = "https:" + cleaned_url[5:]
        elif cleaned_url.startswith("//"):
            cleaned_url = "https:" + cleaned_url
        elif not cleaned_url.startswith("https:"):
            if '.' in cleaned_url:
                cleaned_url = "https://" + cleaned_url
            else:
                return cleaned_url
        cleaned_url = cleaned_url.replace('\\', '/').replace('"', '%22').replace(' ', '%20')
        return cleaned_url

    def _compute_price_metrics(self, base_price, compare_price):
        try:
            base_price = float(base_price)
            compare_price = float(compare_price)
        except (ValueError, TypeError):
            return (None, None)
        if base_price > 0 and compare_price > 0 and abs(compare_price - base_price) < base_price * 5:
            diff = compare_price - base_price
            percent = round((diff / base_price) * 100, 2)
            return (diff, percent)
        return (None, None)

    def _create_minimal_error_dataframe(self, error_message: str) -> pd.DataFrame:
        required_columns = self.excel_settings["required_columns"]
        error_data = {col: [""] for col in required_columns}
        if "상품명" in required_columns:
            error_data["상품명"] = [f"Error: {error_message}"]
        if "판매단가(V포함)" in required_columns:
            error_data["판매단가(V포함)"] = [0]
        if "상품Code" in required_columns:
            error_data["상품Code"] = ["ERROR"]
        return pd.DataFrame(error_data)

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        required_columns = self.excel_settings["required_columns"]
        renamed_columns = {}
        for col in df_copy.columns:
            cleaned_col = str(col).strip()
            if cleaned_col != col:
                renamed_columns[col] = cleaned_col
        if renamed_columns:
            df_copy = df_copy.rename(columns=renamed_columns)
            self.logger.info(f"Cleaned column names: {renamed_columns}")
        column_mapping = {
            "Code": "상품Code",
            "상품코드": "상품Code",
            "상품 코드": "상품Code",
            "상품번호": "상품Code",
            "상품이름": "상품명",
            "상품 이름": "상품명",
            "제품명": "상품명",
            "판매가(V포함)": "판매단가(V포함)",
            "판매가(VAT포함)": "판매단가(V포함)",
            "판매가": "판매단가(V포함)",
            "가격": "판매단가(V포함)",
            "가격(V포함)": "판매단가(V포함)",
            "본사링크": "본사상품링크",
            "상품링크": "본사상품링크",
            "URL": "본사상품링크",
            "수량": "기본수량(1)",
            "기본수량": "기본수량(1)",
            "이미지": "본사 이미지",
            "이미지URL": "본사 이미지",
            "제품이미지": "본사 이미지",
            "상품이미지": "본사 이미지"
        }
        self.logger.debug("Starting column remapping...")
        for src_col, target_col in column_mapping.items():
            if src_col in df_copy.columns and target_col not in df_copy.columns:
                df_copy[target_col] = df_copy[src_col]
                self.logger.info(f"Mapped column '{src_col}' to '{target_col}'")
            elif src_col in df_copy.columns and target_col in df_copy.columns:
                self.logger.debug(f"Column '{target_col}' already exists, skipping mapping from '{src_col}'.")
        for col in required_columns:
            if col not in df_copy.columns:
                self.logger.warning(f"Required column '{col}' is missing. Attempting to find similar.")
                similar_col = self._find_similar_column(df_copy, col)
                if similar_col:
                    self.logger.info(f"Found similar column '{similar_col}' mapped to '{col}'")
                    df_copy[col] = df_copy[similar_col]
                else:
                    self.logger.warning(f"Could not find similar column for '{col}'. Creating default values.")
                    if "단가" in col or "가격" in col:
                        df_copy[col] = 0
                    elif "Code" in col or "코드" in col:
                        df_copy[col] = df_copy.index.map(lambda i: f"GEN-{i}")
                    elif "이미지" in col:
                        df_copy[col] = ""
                    elif "링크" in col:
                        df_copy[col] = ""
                    else:
                        df_copy[col] = ""
        if "source" not in df_copy.columns:
            df_copy["source"] = "haeoreum"
            self.logger.info("Added 'source' column with value 'haeoreum'")
        return df_copy

    def _find_similar_column(self, df: pd.DataFrame, target_column: str):
        column_mapping = {
            "상품명": ["품명", "제품명", "상품", "product", "name", "item", "품목", "상품이름"],
            "판매단가(V포함)": ["단가", "판매가", "가격", "price"],
            "상품Code": ["코드", "code", "item code", "product code"],
            "본사 이미지": ["이미지", "image", "상품이미지"],
            "본사상품링크": ["링크", "link", "url"]
        }
        for col in df.columns:
            if col.lower() == target_column.lower():
                return col
        if target_column in column_mapping:
            for similar in column_mapping[target_column]:
                for col in df.columns:
                    if similar.lower() in col.lower():
                        return col
        return ""
    
    def read_excel_file(self, file_path: str) -> pd.DataFrame:
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.error(f"Excel file not found: {file_path}")
                return self._create_minimal_error_dataframe("File not found")
            self.logger.info(f"Reading Excel file: {file_path}")
            sheet_name = self.excel_settings["sheet_name"]
            try:
                excel_file = pd.ExcelFile(file_path, engine="openpyxl")
                if sheet_name not in excel_file.sheet_names and excel_file.sheet_names:
                    self.logger.warning(f"Sheet '{sheet_name}' not found. Using first sheet: '{excel_file.sheet_names[0]}'")
                    sheet_name = excel_file.sheet_names[0]
                excel_file.close()
            except Exception as e:
                self.logger.warning(f"Could not pre-inspect Excel structure: {str(e)}. Will attempt read with default/first sheet.")
            best_df = None
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, engine="openpyxl")
                if not df.empty:
                    self.logger.info(f"Successfully read {len(df)} rows from sheet '{sheet_name}'.")
                    best_df = df
                else:
                    self.logger.warning(f"Empty DataFrame read from sheet '{sheet_name}'.")
            except Exception as e:
                self.logger.warning(f"Primary read attempt failed: {str(e)}")
            if best_df is None:
                self.logger.info("Primary read failed, attempting alternative strategies...")
                # Alternative strategies can be added here
                best_df = df  # placeholder for alternative read
            if best_df is None or best_df.empty:
                self.logger.error(f"Could not read valid data from {file_path}.")
                return self._create_minimal_error_dataframe("No valid data")
            self.logger.info("Ensuring required columns...")
            processed_df = self._ensure_required_columns(best_df)
            self.logger.info("Finished ensuring columns.")
            return processed_df
        except Exception as e:
            self.logger.error(f"Critical error reading Excel file: {str(e)}", exc_info=True)
            return self._create_minimal_error_dataframe(f"Critical Error: {str(e)}") 