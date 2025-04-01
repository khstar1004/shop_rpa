import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Protection
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import logging
import numpy as np # For checking NaN
import re

from core.data_models import ProcessingResult, MatchResult

logger = logging.getLogger(__name__)

# --- Constants ---
# Define column names based on input/output specs for consistency
# These should match the expected headers in the input and desired output
PRIMARY_REPORT_HEADERS = [
    '구분', '담당자', '업체명', '업체코드', '상품Code', '중분류카테고리', '상품명', 
    '기본수량(1)', '판매단가(V포함)', '본사상품링크', 
    '기본수량(2)', '판매가(V포함)(2)', '판매단가(V포함)(2)', '가격차이(2)', '가격차이(2)%', '고려기프트상품링크', 
    '기본수량(3)', '판매단가(V포함)(3)', '가격차이(3)', '가격차이(3)%', '공급사명', '네이버쇼핑 링크', '공급사 상품링크',
    '본사 이미지', '고려기프트 이미지', '네이버 이미지',
    '매칭_텍스트유사도(고려)', '매칭_이미지유사도(고려)', '매칭_종합유사도(고려)',
    '매칭_텍스트유사도(네이버)', '매칭_이미지유사도(네이버)', '매칭_종합유사도(네이버)',
    '오류 정보'
]

SECONDARY_REPORT_HEADERS = [ # Should match the input file's structure closely
    '구분', '담당자', '업체명', '업체코드', '상품Code', '중분류카테고리', '상품명', 
    '기본수량(1)', '판매단가(V포함)', '본사상품링크', 
    '기본수량(2)', '판매가(V포함)(2)', '판매단가(V포함)(2)', '가격차이(2)', '가격차이(2)%', '고려기프트상품링크', 
    '기본수량(3)', '판매단가(V포함)(3)', '가격차이(3)', '가격차이(3)%', '공급사명', '네이버쇼핑 링크', '공급사 상품링크',
    '본사 이미지', '고려기프트 이미지', '네이버 이미지' # Links only
]

# --- Helper Functions ---

def _safe_get(data: Dict, key: str, default: Any = '') -> Any:
    """Safely get value from dict, handling potential None or NaN."""
    val = data.get(key, default)
    if pd.isna(val):
        return default
    return val

def _format_percentage(value: Optional[float]) -> str:
    """Formats percentage value, handling None or errors."""
    if value is None or not isinstance(value, (int, float)) or pd.isna(value):
        return ''
    return f"{value:.2f}%"

def _format_currency(value: Optional[float]) -> Any:
    """Formats currency value, handling None or errors."""
    if value is None or not isinstance(value, (int, float)) or pd.isna(value):
        return ''
    return value # Return the number itself for openpyxl formatting

def _apply_common_formatting(worksheet):
    """Applies common header and column width formatting."""
    header_font = Font(bold=True)
    center_align = Alignment(horizontal='center', vertical='center')
    
    # Format header
    for cell in worksheet[1]:
        cell.font = header_font
        cell.alignment = center_align

    # Adjust column widths
    for col_idx, column_cells in enumerate(worksheet.columns):
        max_length = 0
        column_letter = get_column_letter(col_idx + 1)
        for cell in column_cells:
            try:
                if cell.value:
                    cell_len = len(str(cell.value))
                    if cell_len > max_length:
                        max_length = cell_len
            except:
                pass
        # Basic width calculation, capped at 50 to prevent excessively wide columns
        adjusted_width = min( (max_length + 2) * 1.1, 50) 
        # Minimum width to ensure headers are readable
        adjusted_width = max(adjusted_width, 10) 
        worksheet.column_dimensions[column_letter].width = adjusted_width
        
    # Freeze top row
    worksheet.freeze_panes = 'A2'

def _apply_conditional_formatting(worksheet, col_map: Dict[str, int]):
    """Applies conditional formatting (yellow fill) and number formats."""
    yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
    currency_format = '#,##0' # Format as integer currency
    percent_format = '0.00%'

    # Get column indices, default to -1 if not found
    koryo_price_diff_idx = col_map.get('가격차이(2)', -1)
    koryo_percent_diff_idx = col_map.get('가격차이(2)%', -1)
    naver_price_diff_idx = col_map.get('가격차이(3)', -1)
    naver_percent_diff_idx = col_map.get('가격차이(3)%', -1)
    
    # Other price columns for currency formatting
    haeoreum_price_idx = col_map.get('판매단가(V포함)', -1)
    koryo_price_idx = col_map.get('판매단가(V포함)(2)', -1)
    naver_price_idx = col_map.get('판매단가(V포함)(3)', -1)

    for row in worksheet.iter_rows(min_row=2):
        # Apply yellow fill for negative price differences
        if koryo_price_diff_idx != -1:
            cell = row[koryo_price_diff_idx]
            if isinstance(cell.value, (int, float)) and cell.value < 0:
                cell.fill = yellow_fill
            # Apply number format only if it's actually a number
            if isinstance(cell.value, (int, float)):
                cell.number_format = currency_format
            # Clear non-numeric strings that shouldn't be there (except error messages)
            elif isinstance(cell.value, str) and not cell.value in ['동일상품 없음', '매칭 실패']:
                cell.value = ''

        if naver_price_diff_idx != -1:
            cell = row[naver_price_diff_idx]
            if isinstance(cell.value, (int, float)) and cell.value < 0:
                cell.fill = yellow_fill
            if isinstance(cell.value, (int, float)):
                cell.number_format = currency_format
            elif isinstance(cell.value, str) and not cell.value in ['동일상품 없음', '매칭 실패']:
                cell.value = ''
            
        # Format percentages
        if koryo_percent_diff_idx != -1:
             cell = row[koryo_percent_diff_idx]
             if isinstance(cell.value, (int, float)):
                 cell.value = cell.value / 100 # Convert to decimal for percentage format
                 cell.number_format = percent_format
             elif isinstance(cell.value, str) and not cell.value in ['동일상품 없음', '매칭 실패']:
                 cell.value = ''

        if naver_percent_diff_idx != -1:
             cell = row[naver_percent_diff_idx]
             if isinstance(cell.value, (int, float)):
                 cell.value = cell.value / 100 # Convert to decimal for percentage format
                 cell.number_format = percent_format
             elif isinstance(cell.value, str) and not cell.value in ['동일상품 없음', '매칭 실패']:
                 cell.value = ''
                 
        # Format currency columns
        for idx in [haeoreum_price_idx, koryo_price_idx, naver_price_idx]:
             if idx != -1:
                 cell = row[idx]
                 if isinstance(cell.value, (int, float)):
                     cell.number_format = currency_format
                 elif isinstance(cell.value, str) and cell.value.strip() and not cell.value == '-':
                     # Try to convert string to number
                     try:
                         cleaned = re.sub(r'[^\d.]', '', cell.value)
                         if cleaned:
                             cell.value = float(cleaned)
                             cell.number_format = currency_format
                         else:
                             cell.value = ''
                     except (ValueError, TypeError):
                         cell.value = ''


# --- Primary Report Generation ---

def generate_primary_report(
    results: List[ProcessingResult], 
    config: Dict[str, Any], 
    start_time: datetime, 
    end_time: datetime
) -> Optional[str]:
    """
    Generates the Primary (1차) Excel report with detailed comparison results.
    Includes all original columns plus matching details and image links.
    """
    output_dir = config['PATHS']['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"RPA_1차_결과_{timestamp}.xlsx")
    
    report_data = []

    for result in results:
        source_data = result.source_product.original_input_data
        row = {}
        
        # --- Populate Original Data ---
        # Use PRIMARY_REPORT_HEADERS to ensure all needed original columns are included
        for header in PRIMARY_REPORT_HEADERS:
             # Check if header exists in original data
             if header in source_data:
                 row[header] = _safe_get(source_data, header)
             else:
                 # Initialize missing original columns expected in the output
                 row[header] = '' 
                 
        # Ensure core identifiers are present even if missing in original_input_data (e.g. from parsing)
        row['상품명'] = result.source_product.name
        row['상품Code'] = result.source_product.id
        row['판매단가(V포함)'] = result.source_product.price # Use parsed price
        row['본사 이미지'] = _safe_get(source_data, '본사 이미지', result.source_product.image_url)
        row['본사상품링크'] = _safe_get(source_data, '본사상품링크', result.source_product.url)

        # --- Populate Koryo Comparison Data ---
        k_match = result.best_koryo_match
        if k_match:
            row['기본수량(2)'] = '' # Not available from scraper? Assume same as source or leave blank
            row['판매가(V포함)(2)'] = '' # Not directly scraped, maybe calculate if needed?
            row['판매단가(V포함)(2)'] = k_match.matched_product.price
            row['가격차이(2)'] = k_match.price_difference
            row['가격차이(2)%'] = k_match.price_difference_percent
            row['고려기프트상품링크'] = k_match.matched_product.url
            row['고려기프트 이미지'] = k_match.matched_product.image_url
            row['매칭_텍스트유사도(고려)'] = f"{k_match.text_similarity:.3f}"
            row['매칭_이미지유사도(고려)'] = f"{k_match.image_similarity:.3f}"
            row['매칭_종합유사도(고려)'] = f"{k_match.combined_similarity:.3f}"
        else:
             # Fill Koryo columns with blanks or '없음' if no match found
             row['가격차이(2)'] = '동일상품 없음' # Indicate no match found
             # Clear other related Koryo fields
             for col in ['기본수량(2)', '판매가(V포함)(2)', '판매단가(V포함)(2)', 
                         '가격차이(2)%', '고려기프트상품링크', '고려기프트 이미지', 
                         '매칭_텍스트유사도(고려)', '매칭_이미지유사도(고려)', '매칭_종합유사도(고려)']:
                 if col not in row or row[col] == '': # Avoid overwriting existing blanks
                      row[col] = '' 

        # --- Populate Naver Comparison Data ---
        n_match = result.best_naver_match
        if n_match:
            row['기본수량(3)'] = '' # Not available from Naver API?
            # 판매단가(V포함)(3) is the scraped 'lprice'
            row['판매단가(V포함)(3)'] = n_match.matched_product.price 
            row['가격차이(3)'] = n_match.price_difference
            row['가격차이(3)%'] = n_match.price_difference_percent
            row['공급사명'] = n_match.matched_product.brand or _safe_get(n_match.matched_product.original_input_data, 'mallName', '') # Use brand or mallName
            row['네이버쇼핑 링크'] = n_match.matched_product.url
            row['공급사 상품링크'] = '' # Usually same as Naver link from API
            row['네이버 이미지'] = n_match.matched_product.image_url
            row['매칭_텍스트유사도(네이버)'] = f"{n_match.text_similarity:.3f}"
            row['매칭_이미지유사도(네이버)'] = f"{n_match.image_similarity:.3f}"
            row['매칭_종합유사도(네이버)'] = f"{n_match.combined_similarity:.3f}"
        else:
             # Fill Naver columns with blanks or '없음'
             row['가격차이(3)'] = '동일상품 없음' # Indicate no match found
             # Clear other related Naver fields
             for col in ['기본수량(3)', '판매단가(V포함)(3)', '가격차이(3)%', 
                         '공급사명', '네이버쇼핑 링크', '공급사 상품링크', '네이버 이미지',
                         '매칭_텍스트유사도(네이버)', '매칭_이미지유사도(네이버)', '매칭_종합유사도(네이버)']:
                 if col not in row or row[col] == '':
                     row[col] = ''

        # Add error info if processing failed for this item
        row['오류 정보'] = result.error if result.error else ''
        
        report_data.append(row)

    if not report_data:
        logger.warning("No data to generate primary report.")
        return None

    try:
        # Create DataFrame with specified column order
        df = pd.DataFrame(report_data)
        # Reorder columns according to PRIMARY_REPORT_HEADERS, adding missing ones if needed
        df = df.reindex(columns=PRIMARY_REPORT_HEADERS, fill_value='') 

        wb = Workbook()
        ws = wb.active
        ws.title = "RPA 1차 결과"
        
        # Add header row for processing times
        ws['A1'] = f"RPA 처리 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ws['D1'] = f"RPA 처리 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ws.merge_cells('A1:C1')
        ws.merge_cells('D1:F1')
        
        # Write DataFrame to Excel starting from row 2
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 2):
            for c_idx, value in enumerate(row, 1):
                ws.cell(row=r_idx, column=c_idx, value=value)
        
        # Create column name to index map for formatting (adjust for extra time header row)
        col_map = {header: idx + 1 for idx, header in enumerate(df.columns)}
        
        _apply_common_formatting(ws) 
        _apply_conditional_formatting(ws, col_map)

        wb.save(output_filename)
        logger.info(f"Primary report successfully generated: {output_filename}")
        return output_filename

    except Exception as e:
        logger.error(f"Failed to generate primary Excel report: {e}", exc_info=True)
        return None

# --- Secondary Report Generation (with Filtering) ---

def generate_secondary_report(
    results: List[ProcessingResult], 
    config: Dict[str, Any],
    input_filepath: str
) -> Optional[str]:
    """
    Generates the Secondary (2차) Excel report after applying filtering rules.
    Contains only rows with negative price differences that meet criteria.
    Outputs columns matching the input structure, with image links only.
    """
    output_dir = config['PATHS']['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename based on input filename
    base_name = os.path.splitext(os.path.basename(input_filepath))[0]
    output_filename = os.path.join(output_dir, f"{base_name}-result.xlsx")

    filtered_data = []

    for result in results:
        if result.error: # Skip errored items for secondary report
            continue
            
        source_data = result.source_product.original_input_data
        
        # --- Apply Filtering Logic ---
        k_match = result.best_koryo_match
        n_match = result.best_naver_match
        
        # Check initial condition: must have at least one negative price difference
        koryo_is_cheaper = k_match and k_match.price_difference < 0
        naver_is_cheaper = n_match and n_match.price_difference < 0
        
        if not (koryo_is_cheaper or naver_is_cheaper):
            continue # Rule 1: Must have at least one competitor cheaper

        # --- Filter Koryo Match (if cheaper) ---
        keep_koryo = False
        if koryo_is_cheaper:
            # Rule 3 (Positive Diff): Already checked by koryo_is_cheaper
            # Rule: Promo Site 1% Threshold (주의사항 1, #3)
            # Koryo is considered a promotional site by default based on context
            if k_match.price_difference_percent <= -1.0: # Must be at least 1% cheaper
                 keep_koryo = True
            else:
                 logger.debug(f"Filtering Koryo match for {result.source_product.name}: Price diff {k_match.price_difference_percent:.2f}% <= -1.0% is False.")

        # --- Filter Naver Match (if cheaper) ---
        keep_naver = False
        if naver_is_cheaper:
            # Rule 3 (Positive Diff): Already checked by naver_is_cheaper
            is_promo = n_match.matched_product.is_promotional_site
            naver_basic_qty_present = True # Assume true unless we parse it (not currently done)
            
            # Rule: Promo Site 1% Threshold (주의사항 1, #3)
            if is_promo:
                if n_match.price_difference_percent <= -1.0: # Must be at least 1% cheaper
                    keep_naver = True
                else:
                    logger.debug(f"Filtering Naver Promo match for {result.source_product.name}: Price diff {n_match.price_difference_percent:.2f}% <= -1.0% is False.")
            # Rule: General Site 10% Threshold (주의사항 1, #4)
            elif not is_promo:
                if n_match.price_difference_percent <= -10.0: # Must be at least 10% cheaper
                    keep_naver = True
                else:
                     # Rule 2 (Naver Basic Qty Check - partially applied)
                     # If price diff is between 0% and -10% for general site, filter it out
                     # (This covers the "<= 10%" removal rule, interpreting it as "not cheap enough")
                     logger.debug(f"Filtering Naver General match for {result.source_product.name}: Price diff {n_match.price_difference_percent:.2f}% <= -10.0% is False.")
            
        # Rule 4 & 주의사항 2, #1: Remove row if no valid cheaper competitor remains after filtering
        if not keep_koryo and not keep_naver:
             logger.debug(f"Removing row for {result.source_product.name}: No valid cheaper competitor after filtering.")
             continue
             
        # --- Construct Row Data for Secondary Report ---
        row = {}
        # Copy all original data first
        for header in SECONDARY_REPORT_HEADERS:
             row[header] = _safe_get(source_data, header)
             
        # Override specific fields with filtered match data
        
        # Koryo Data (only if kept)
        if keep_koryo:
             row['기본수량(2)'] = '' # Keep original or blank
             row['판매가(V포함)(2)'] = '' # Keep original or blank
             row['판매단가(V포함)(2)'] = k_match.matched_product.price
             row['가격차이(2)'] = k_match.price_difference
             row['가격차이(2)%'] = k_match.price_difference_percent
             row['고려기프트상품링크'] = k_match.matched_product.url
             row['고려기프트 이미지'] = k_match.matched_product.image_url # Keep image link
        else:
             # Clear Koryo comparison fields if Koryo was filtered out
             for col in ['기본수량(2)', '판매가(V포함)(2)', '판매단가(V포함)(2)', 
                         '가격차이(2)', '가격차이(2)%', '고려기프트상품링크', '고려기프트 이미지']:
                 if col in row: row[col] = ''
                 
        # Naver Data (only if kept)
        if keep_naver:
             row['기본수량(3)'] = '' # Keep original or blank
             row['판매단가(V포함)(3)'] = n_match.matched_product.price 
             row['가격차이(3)'] = n_match.price_difference
             row['가격차이(3)%'] = n_match.price_difference_percent
             row['공급사명'] = n_match.matched_product.brand or _safe_get(n_match.matched_product.original_input_data, 'mallName', '')
             row['네이버쇼핑 링크'] = n_match.matched_product.url
             row['공급사 상품링크'] = '' # Keep original or blank
             row['네이버 이미지'] = n_match.matched_product.image_url # Keep image link
        else:
             # Clear Naver comparison fields if Naver was filtered out
             for col in ['기본수량(3)', '판매단가(V포함)(3)', '가격차이(3)', '가격차이(3)%', 
                         '공급사명', '네이버쇼핑 링크', '공급사 상품링크', '네이버 이미지']:
                 if col in row: row[col] = ''
                 
        # Ensure essential source data is preserved
        row['구분'] = _safe_get(source_data, '구분')
        row['담당자'] = _safe_get(source_data, '담당자')
        row['상품명'] = result.source_product.name
        row['상품Code'] = result.source_product.id
        row['판매단가(V포함)'] = result.source_product.price
        row['본사 이미지'] = _safe_get(source_data, '본사 이미지', result.source_product.image_url)
        row['본사상품링크'] = _safe_get(source_data, '본사상품링크', result.source_product.url)

        filtered_data.append(row)

    if not filtered_data:
        logger.warning("No data remaining after filtering for secondary report.")
        # Still create an empty file? Or return None? Let's create an empty file for consistency.
        # return None 
        df = pd.DataFrame(columns=SECONDARY_REPORT_HEADERS)
    else:
         # Create DataFrame with specified column order
         df = pd.DataFrame(filtered_data)
         df = df.reindex(columns=SECONDARY_REPORT_HEADERS, fill_value='') 

    logger.info(f"Generating secondary report with {len(df)} rows after filtering.")

    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "필터링 결과" # Sheet name for secondary report

        # Write DataFrame to Excel
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                 # Ensure links are treated as strings
                 if isinstance(value, str) and value.startswith('http'):
                     ws.cell(row=r_idx, column=c_idx, value=value).style = 'Hyperlink' 
                 else:
                     ws.cell(row=r_idx, column=c_idx, value=value)
        
        # Create column map for formatting
        col_map = {header: idx + 1 for idx, header in enumerate(df.columns)}

        _apply_common_formatting(ws)
        _apply_conditional_formatting(ws, col_map) # Apply yellow fill and number formats

        wb.save(output_filename)
        logger.info(f"Secondary report successfully generated: {output_filename}")
        return output_filename

    except Exception as e:
        logger.error(f"Failed to generate secondary Excel report: {e}", exc_info=True)
        return None

# Remove old generate_report function if it exists
# def generate_report(...): pass