import os
import re
import logging
import pandas as pd


class ExcelConverter:
    def __init__(self, config: dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
    
    def convert_xls_to_xlsx(self, input_directory: str) -> str:
        """
        Converts the first XLS file found in the input directory to XLSX format.
        Returns the path to the converted file, or empty string if no conversion was done.
        """
        try:
            self.logger.info(f"Looking for XLS files in: {input_directory}")
            xls_files = [f for f in os.listdir(input_directory) if f.lower().endswith(".xls")]
            
            if not xls_files:
                self.logger.warning("No XLS files found in the directory.")
                return ""
            
            file_name = xls_files[0]
            file_path = os.path.join(input_directory, file_name)
            self.logger.info(f"Converting XLS file: {file_path}")
            
            # Use pandas to read the XLS file
            try:
                # First try with default encoding
                tables = pd.read_html(file_path)
            except Exception:
                # If that fails, try with Korean encoding
                self.logger.info("Retrying with Korean encoding (cp949)")
                tables = pd.read_html(file_path, encoding="cp949")
            
            if not tables:
                self.logger.error("No tables found in the XLS file.")
                return ""
            
            df = tables[0]
            
            # Clean up the dataframe
            # Assume first row contains headers
            if not all(isinstance(col, str) for col in df.columns):
                df.columns = df.iloc[0].astype(str).str.strip()
                df = df.drop(0)
            
            # Clean product names if they exist
            if "상품명" in df.columns:
                pattern = r"(\d{4}_[A-Z]\.)|(\d+\+\d+)|[^a-zA-Z0-9가-힣\s]|\s+"
                df["상품명"] = df["상품명"].apply(
                    lambda x: re.sub(pattern, " ", str(x)).strip() if pd.notna(x) else x
                )
            
            # Add empty image columns if they don't exist
            for column in ["본사 이미지", "고려기프트 이미지", "네이버 이미지"]:
                if column not in df.columns:
                    df[column] = ""
            
            # Save to XLSX format
            output_file_name = file_name.replace(".xls", ".xlsx")
            output_file_path = os.path.join(input_directory, output_file_name)
            
            df.to_excel(output_file_path, index=False)
            self.logger.info(f"Converted file saved to: {output_file_path}")
            
            return output_file_path
        except Exception as e:
            self.logger.error(f"Error converting XLS to XLSX: {str(e)}", exc_info=True)
            return ""
    
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