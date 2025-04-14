import os
import re
import logging
import pandas as pd
from openpyxl import load_workbook


class ExcelPostProcessor:
    def __init__(self, config: dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
    
    def remove_at_symbol(self, file_path: str) -> str:
        """
        Removes @ symbols from all cells in an Excel file.
        Returns the path to the processed file.
        """
        try:
            self.logger.info(f"Removing @ symbols from: {file_path}")
            
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return file_path
            
            # Create a temporary file path
            temp_dir = os.path.dirname(file_path)
            temp_file = os.path.join(temp_dir, f"temp_{os.path.basename(file_path)}")
            
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Remove @ symbols from all string columns
            for column in df.columns:
                if df[column].dtype == 'object':  # Object dtype typically means strings
                    df[column] = df[column].astype(str).str.replace('@', '', regex=False)
            
            # Special handling for image columns
            for col in ['본사 이미지', '고려기프트 이미지', '네이버 이미지']:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.replace('@', '') if isinstance(x, str) else x
                    )
            
            # Save to the temporary file
            df.to_excel(temp_file, index=False)
            
            # Replace the original file with the cleaned version
            os.remove(file_path)
            os.rename(temp_file, file_path)
            
            self.logger.info(f"Successfully removed @ symbols from {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error removing @ symbols: {str(e)}", exc_info=True)
            return file_path
    
    def _fix_image_formulas(self, file_path: str) -> None:
        """
        Fixes IMAGE formulas in Excel files by ensuring proper format and removing any
        problematic characters.
        """
        try:
            wb = load_workbook(file_path)
            made_changes = False
            
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                
                # Scan all cells for IMAGE formulas
                for row in ws.iter_rows():
                    for cell in row:
                        if cell.value and isinstance(cell.value, str) and cell.value.startswith("=IMAGE("):
                            original = cell.value
                            
                            # Fix common issues in IMAGE formulas
                            fixed = original
                            
                            # 1. Remove @ symbols
                            fixed = fixed.replace('@', '')
                            
                            # 2. Replace backslashes with forward slashes
                            fixed = fixed.replace('\\', '/')
                            
                            # 3. Ensure proper quotation marks
                            if '"' not in fixed:
                                fixed = fixed.replace('=IMAGE(', '=IMAGE("')
                                if fixed.endswith(')'):
                                    fixed = fixed[:-1] + '")'
                                else:
                                    # Add missing closing parenthesis if needed
                                    fixed = fixed + '")'
                            
                            # 4. Fix double quotes
                            fixed = fixed.replace('""', '"')
                            
                            # 5. Ensure proper scaling mode parameter
                            if ',2)' not in fixed and ')' in fixed:
                                fixed = fixed.replace(')', ',2)')
                            
                            # Apply changes if formula was modified
                            if fixed != original:
                                cell.value = fixed
                                made_changes = True
            
            if made_changes:
                wb.save(file_path)
                self.logger.info(f"Image formulas fixed in {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error fixing image formulas: {str(e)}", exc_info=True)
    
    def post_process_excel_file(self, file_path: str) -> str:
        """
        Performs all post-processing steps on an Excel file:
        1. Removes @ symbols
        2. Fixes IMAGE formulas
        3. Other cleanup as needed
        
        Returns the path to the processed file.
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found for post-processing: {file_path}")
                return file_path
            
            # First, remove @ symbols
            cleaned_file = self.remove_at_symbol(file_path)
            
            # Then fix image formulas
            try:
                self._fix_image_formulas(cleaned_file)
            except Exception as img_err:
                self.logger.error(f"Error fixing image formulas: {str(img_err)}")
            
            self.logger.info(f"Post-processing complete for {cleaned_file}")
            return cleaned_file
            
        except Exception as e:
            self.logger.error(f"Error during post-processing: {str(e)}", exc_info=True)
            return file_path 