import logging
import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


class ExcelFormatter:
    def __init__(self, config: dict, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.bold_font = Font(bold=True)
        self.center_align = Alignment(horizontal="center", vertical="center")
        self.header_fill = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")
        self.thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin")
        )
        self.alt_row_fill = PatternFill(start_color="F5F5F5", end_color="F5F5F5", fill_type="solid")
        self.image_row_height = 110
        self.row_height = 18
        
        # Additional fill styles for specific formatting
        self.hyperlink_font = Font(color="0000FF", underline="single")
        self.highlight_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        self.header_font = Font(bold=True, size=11)

    def apply_formatting_to_excel(self, excel_file: str) -> None:
        try:
            wb = load_workbook(excel_file)
            ws = wb.active
            # Format header row
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col)
                cell.font = self.bold_font
                cell.alignment = self.center_align
                cell.fill = self.header_fill
                cell.border = self.thin_border
            # Identify image and status columns
            image_col_indices = []
            status_col_indices = []
            for col in range(1, ws.max_column + 1):
                header = ws.cell(row=1, column=col).value
                if header and "이미지" in str(header):
                    image_col_indices.append(col)
                    col_letter = get_column_letter(col)
                    ws.column_dimensions[col_letter].width = 20
                elif header and ("매칭_상황" in str(header) or "텍스트유사도" in str(header)):
                    status_col_indices.append(col)
                    col_letter = get_column_letter(col)
                    ws.column_dimensions[col_letter].width = 30
            # Apply zebra striping and row height adjustments
            for row in range(2, ws.max_row + 1):
                has_image = any(
                    ws.cell(row=row, column=col).value and
                    isinstance(ws.cell(row=row, column=col).value, str) and
                    (ws.cell(row=row, column=col).value.startswith("http") or ws.cell(row=row, column=col).value.startswith("=IMAGE"))
                    for col in image_col_indices
                )
                has_error = any(
                    ws.cell(row=row, column=col).value and
                    isinstance(ws.cell(row=row, column=col).value, str) and
                    len(ws.cell(row=row, column=col).value) > 10
                    for col in status_col_indices
                )
                ws.row_dimensions[row].height = self.image_row_height if has_image else (60 if has_error else self.row_height)
                if row % 2 == 0:
                    for col in range(1, ws.max_column + 1):
                        cell = ws.cell(row=row, column=col)
                        if not cell.fill or cell.fill.start_color.index == "FFFFFF":
                            cell.fill = self.alt_row_fill
                for col in range(1, ws.max_column + 1):
                    cell = ws.cell(row=row, column=col)
                    cell.border = self.thin_border
                    if col in status_col_indices and cell.value and isinstance(cell.value, str) and "상품이 없음" in cell.value:
                        cell.font = Font(color="FF0000", bold=True)
                        cell.alignment = Alignment(wrap_text=True, vertical="center")
            # Adjust column widths
            for col in range(1, ws.max_column + 1):
                if col in status_col_indices or col in image_col_indices:
                    continue
                max_length = 0
                for row in range(1, ws.max_row + 1):
                    cell_val = ws.cell(row=row, column=col).value
                    if cell_val:
                        max_length = max(max_length, len(str(cell_val)))
                adjusted_width = min(max_length + 4, 50)
                header_val = str(ws.cell(row=1, column=col).value)
                if "상품명" in header_val:
                    adjusted_width = max(adjusted_width, 35)
                elif "링크" in header_val:
                    adjusted_width = max(adjusted_width, 25)
                else:
                    adjusted_width = max(adjusted_width, 12)
                ws.column_dimensions[get_column_letter(col)].width = adjusted_width
            ws.freeze_panes = "A2"
            wb.save(excel_file)
            self.logger.info(f"Enhanced formatting applied to {excel_file}")
        except Exception as e:
            self.logger.error(f"Error applying formatting: {str(e)}", exc_info=True)

    def add_hyperlinks_to_excel(self, file_path: str) -> str:
        """
        Add hyperlinks to URLs in the Excel file.
        
        This function:
        1. Identifies columns containing URLs
        2. Converts text URLs to clickable hyperlinks
        3. Applies formatting to make hyperlinks stand out
        4. Applies additional formatting like zebra striping and headers
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            str: Path to the processed file
        """
        try:
            self.logger.info(f"Adding hyperlinks to Excel file: {file_path}")
            
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return file_path
            
            # Load the workbook
            try:
                df = pd.read_excel(file_path)
                wb = load_workbook(file_path)
                ws = wb.active
            except Exception as e:
                self.logger.error(f"Error loading workbook: {str(e)}")
                return file_path
            
            # Define link columns to process
            link_columns = [
                "본사상품링크", 
                "고려기프트 상품링크", 
                "네이버 쇼핑 링크", 
                "공급사 상품링크",
                "본사링크", 
                "고려 링크", 
                "네이버 링크"
            ]
            
            # Process all columns containing 'link' or 'URL' in their name
            for col in df.columns:
                if '링크' in col or 'link' in col.lower() or 'url' in col.lower():
                    if col not in link_columns:
                        link_columns.append(col)
            
            # Process price difference columns to highlight negative values
            price_diff_cols = [col for col in df.columns if "가격차이" in col and "%" not in col]
            
            # Process each row and column
            for row_idx in range(2, ws.max_row + 1):  # Skip header row
                # First check for price difference highlights
                highlight_row = False
                for col_name in price_diff_cols:
                    if col_name in df.columns:
                        col_idx = list(df.columns).index(col_name) + 1
                        cell = ws.cell(row=row_idx, column=col_idx)
                        try:
                            value = float(cell.value) if cell.value is not None else 0
                            if value < 0:
                                cell.fill = self.highlight_fill
                                highlight_row = True  # Mark for row highlighting
                        except (ValueError, TypeError):
                            pass
                
                # Highlight entire row if needed
                if highlight_row:
                    for col_idx in range(1, ws.max_column + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        if not cell.fill or cell.fill.start_color.index == "FFFFFF":
                            cell.fill = self.highlight_fill
                
                # Process hyperlinks
                for col_name in link_columns:
                    if col_name in df.columns:
                        col_idx = list(df.columns).index(col_name) + 1
                        cell = ws.cell(row=row_idx, column=col_idx)
                        
                        if cell.value and isinstance(cell.value, str):
                            # Clean URL and apply hyperlink
                            url = cell.value.strip()
                            
                            # Skip cells that don't contain valid URLs
                            if not (url.startswith('http://') or url.startswith('https://') or
                                    url.startswith('www.')):
                                continue
                            
                            # Ensure URL starts with https:// or http://
                            if url.startswith('www.'):
                                url = 'https://' + url
                            
                            # Apply hyperlink
                            cell.hyperlink = url
                            cell.font = self.hyperlink_font
                            cell.value = url  # Ensure cell shows the URL text
                            
                            # Set alignment for hyperlink cells
                            cell.alignment = Alignment(vertical='center')
            
            # Set header formatting 
            header_row = ws[1]
            for cell in header_row:
                cell.font = self.header_font
                cell.fill = self.header_fill
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = self.thin_border
            
            # Adjust column widths (ensure links have enough space)
            for col_name in link_columns:
                if col_name in df.columns:
                    col_idx = list(df.columns).index(col_name) + 1
                    col_letter = get_column_letter(col_idx)
                    ws.column_dimensions[col_letter].width = 30
            
            # Set freeze panes at A2 (keep header visible while scrolling)
            ws.freeze_panes = "A2"
            
            # Save the updated workbook
            wb.save(file_path)
            self.logger.info(f"Successfully added hyperlinks to {file_path}")
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error adding hyperlinks to Excel: {str(e)}", exc_info=True)
            return file_path

    def filter_excel_by_price_diff(self, file_path: str) -> str:
        """
        Filter Excel file to show only rows with negative price differences.
        
        This function:
        1. Loads the Excel file
        2. Filters rows with negative price differences
        3. Creates a new Excel file with only those rows
        4. Applies formatting to the new file
        
        Args:
            file_path: Path to the input Excel file
            
        Returns:
            str: Path to the filtered Excel file
        """
        try:
            from datetime import datetime
            
            self.logger.info(f"Filtering Excel file by price differences: {file_path}")
            
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return file_path
            
            # Generate output filename
            input_basename = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_basename}_upload_{timestamp}.xlsx"
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(file_path), 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Load Excel file
            df = pd.read_excel(file_path)
            
            # Helper function to convert values to float
            def to_float(value):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            # Convert price difference columns to numeric values
            price_diff_columns = [col for col in df.columns if "가격차이" in col and "%" not in col]
            for col in price_diff_columns:
                df[col] = df[col].apply(to_float)
            
            # Build filter condition for negative price differences
            if not price_diff_columns:
                self.logger.warning("No price difference columns found. Returning original file.")
                return file_path
            
            # Filter rows with any negative price difference
            filter_condition = False
            for col in price_diff_columns:
                filter_condition |= (df[col] < 0)
            
            filtered_df = df[filter_condition]
            
            if filtered_df.empty:
                self.logger.warning("No rows with negative price differences found. Returning original file.")
                return file_path
            
            # Create mapping for standardized column names
            column_mapping = {
                "구분": "구분(승인관리:A/가격관리:P)",
                "담당자": "담당자",
                "업체명": "공급사명",
                "업체코드": "공급처코드",
                "Code": "상품코드",
                "상품Code": "상품코드",
                "중분류카테고리": "카테고리(중분류)",
                "상품명": "상품명",
                "기본수량(1)": "본사 기본수량",
                "판매단가(V포함)": "판매단가1(VAT포함)",
                "본사상품링크": "본사링크",
                "기본수량(2)": "고려 기본수량",
                "판매단가(V포함)(2)": "판매단가2(VAT포함)",
                "가격차이(2)": "고려 가격차이",
                "가격차이(2)(%)": "고려 가격차이(%)",
                "고려기프트 상품링크": "고려 링크",
                "기본수량(3)": "네이버 기본수량",
                "판매단가(V포함)(3)": "판매단가3(VAT포함)",
                "가격차이(3)": "네이버 가격차이",
                "가격차이(3)(%)": "네이버가격차이(%)",
                "공급사명": "네이버 공급사명",
                "공급사 상품링크": "네이버 링크",
                "본사 이미지": "해오름(이미지링크)",
                "고려기프트 이미지": "고려기프트(이미지링크)",
                "네이버 이미지": "네이버쇼핑(이미지링크)"
            }
            
            # Rename columns if they exist in the dataframe
            rename_mapping = {
                col: new_name for col, new_name in column_mapping.items() 
                if col in filtered_df.columns
            }
            
            filtered_df = filtered_df.rename(columns=rename_mapping)
            
            # Save filtered dataframe to Excel
            filtered_df.to_excel(output_path, index=False)
            
            # Apply formatting to the output file
            wb = load_workbook(output_path)
            ws = wb.active
            
            # Set freeze panes
            ws.freeze_panes = "A2"
            
            # Format headers
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col)
                cell.font = self.header_font
                cell.fill = self.header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = self.thin_border
            
            # Format data rows
            for row in range(2, ws.max_row + 1):
                for col in range(1, ws.max_column + 1):
                    cell = ws.cell(row=row, column=col)
                    cell.border = self.thin_border
                    cell.alignment = Alignment(vertical="center", wrap_text=True)
                    
                    # Center-align numeric values
                    try:
                        float(cell.value)
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    except (ValueError, TypeError):
                        pass
            
            # Set column widths
            for col in range(1, ws.max_column + 1):
                col_letter = get_column_letter(col)
                header = ws.cell(row=1, column=col).value
                
                if header and ("링크" in str(header) or "URL" in str(header)):
                    ws.column_dimensions[col_letter].width = 30
                elif header and "이미지" in str(header):
                    ws.column_dimensions[col_letter].width = 20
                elif header and "상품명" in str(header):
                    ws.column_dimensions[col_letter].width = 35
                else:
                    ws.column_dimensions[col_letter].width = 15
            
            # Save the formatted workbook
            wb.save(output_path)
            
            self.logger.info(f"Successfully filtered and saved file to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error filtering Excel by price differences: {str(e)}", exc_info=True)
            return file_path 