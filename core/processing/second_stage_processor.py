import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def safe_to_numeric(series, errors='coerce'):
    """
    Converts a pandas Series to numeric, handling common non-numeric placeholders.
    """
    non_numeric_placeholders = ['#REF!', '', '-', '동일상품 없음', '(공백)', '(공백 or 없음)', '가격이 범위 내에 없거나 검색된 상품이 없음']
    series = series.replace(non_numeric_placeholders, np.nan)
    # Remove commas for thousands separators if present
    if series.dtype == 'object':
        series = series.str.replace(',', '', regex=False)
    return pd.to_numeric(series, errors=errors)

def process_second_stage(input_file_path, output_dir=None):
    """
    Main function to process the 1st stage Excel file and generate the 2nd stage file
    based on the rules specified in data/inputandoutput.txt
    
    Args:
        input_file_path (str): Path to the 1st stage Excel file
        output_dir (str, optional): Output directory. Defaults to creating a 'final' directory in the same location.
    
    Returns:
        str: Path to the generated 2nd stage file or None if an error occurs
    """
    start_time = datetime.now()
    logger.info(f"Starting 2nd stage processing of {input_file_path}")
    
    if not os.path.exists(input_file_path):
        logger.error(f"Input file not found: {input_file_path}")
        return None
        
    # Determine output directory and filename
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_file_path), "final")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(input_file_path)
    name, ext = os.path.splitext(base_name)
    # Following the pattern in inputandoutput.txt: "{name}-result{ext}"
    output_filename = f"{name}-result{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # Read the input Excel file
        df = pd.read_excel(input_file_path)
        logger.info(f"Read {len(df)} rows from {input_file_path}")
        
        # Apply 2nd stage processing rules
        df_result = apply_second_stage_rules(df)
        
        if df_result.empty:
            logger.warning("No rows remain after applying 2nd stage rules")
            return None
            
        # Save the processed DataFrame to Excel
        df_result.to_excel(output_path, index=False)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"2nd stage processing complete. Saved to {output_path} ({duration:.2f}s)")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing 2nd stage: {str(e)}", exc_info=True)
        return None

def apply_second_stage_rules(df):
    """
    Apply the 2nd stage filtering rules as defined in data/inputandoutput.txt
    
    Args:
        df (pd.DataFrame): The input DataFrame from the 1st stage file
        
    Returns:
        pd.DataFrame: The processed DataFrame after applying all rules
    """
    logger.info("Applying 2nd stage filtering rules...")
    df_processed = df.copy()
    
    # --- 1. Data Conversion ---
    # Convert relevant columns to numeric values
    numeric_cols = {
        '판매단가(V포함)': '본사_판매단가',
        '기본수량(1)': '본사_기본수량',
        '판매단가(V포함)(2)': '고려_판매단가',
        '가격차이(2)': '고려_가격차이',
        '가격차이(2)%': '고려_가격차이_퍼센트',
        '기본수량(2)': '고려_기본수량',
        '판매단가(V포함)(3)': '네이버_판매단가',
        '가격차이(3)': '네이버_가격차이',
        '가격차이(3)%': '네이버_가격차이_퍼센트',
        '기본수량(3)': '네이버_기본수량',
    }
    
    # Rename and convert numeric columns
    for original, new in numeric_cols.items():
        if original in df_processed.columns:
            df_processed[new] = safe_to_numeric(df_processed[original])
        else:
            logger.warning(f"Column '{original}' not found in input data")
    
    # --- 2. Apply Yellow Cell Filter (Rule 1) ---
    # Keep only rows with either Koreagift or Naver price differences that are negative
    yellow_mask = (df_processed['고려_가격차이'] < 0) | (df_processed['네이버_가격차이'] < 0)
    df_processed = df_processed[yellow_mask].copy()
    logger.info(f"After yellow cell filter: {len(df_processed)} rows")
    
    # --- 3. Initialize Validity Flags ---
    df_processed['고려_유효'] = df_processed['고려_가격차이'].notna() & (df_processed['고려_가격차이'] < 0)
    df_processed['네이버_유효'] = df_processed['네이버_가격차이'].notna() & (df_processed['네이버_가격차이'] < 0)
    
    # --- 4. Rule for Koreagift: Invalidate if price difference <= 1% ---
    # Get the absolute percentage and check if it's <= 1%
    korea_invalid_mask = df_processed['고려_가격차이_퍼센트'].notna() & (df_processed['고려_가격차이_퍼센트'].abs() <= 0.01)
    df_processed.loc[korea_invalid_mask, '고려_유효'] = False
    logger.info(f"Koreagift rows invalidated due to price diff <= 1%: {korea_invalid_mask.sum()}")
    
    # --- 5. Rule for Naver: Invalidate if 기본수량 missing and price diff <= 10% ---
    naver_invalid_mask = (
        df_processed['네이버_기본수량'].isna() & 
        df_processed['네이버_가격차이_퍼센트'].notna() & 
        (df_processed['네이버_가격차이_퍼센트'].abs() <= 0.10)
    )
    df_processed.loc[naver_invalid_mask, '네이버_유효'] = False
    logger.info(f"Naver rows invalidated due to missing quantity and price diff <= 10%: {naver_invalid_mask.sum()}")
    
    # --- 6. Remove rows where both Koreagift and Naver data are invalid ---
    both_valid_mask = df_processed['고려_유효'] | df_processed['네이버_유효']
    df_processed = df_processed[both_valid_mask].copy()
    logger.info(f"After removing rows with no valid data: {len(df_processed)} rows")
    
    # --- 7. Clear data for invalidated sections ---
    # Clear Koreagift data if invalid
    korea_cols = [col for col in df_processed.columns if '(2)' in col or '고려' in col]
    for col in korea_cols:
        if col in df_processed.columns:
            df_processed.loc[~df_processed['고려_유효'], col] = np.nan
    
    # Clear Naver data if invalid
    naver_cols = [col for col in df_processed.columns if '(3)' in col or '네이버' in col or col == '공급사명']
    for col in naver_cols:
        if col in df_processed.columns:
            df_processed.loc[~df_processed['네이버_유효'], col] = np.nan
    
    # --- 8. Map columns to output format ---
    column_mapping = {
        '구분': '구분(A/P)',
        '담당자': '담당자',
        '업체명': '공급사명',
        '업체코드': '공급처코드',
        '상품Code': '상품코드',
        '중분류카테고리': '카테고리(중분류)',
        '상품명': '상품명',
        '기본수량(1)': '본사 기본수량',
        '판매단가(V포함)': '판매단가1(VAT포함)',
        '본사상품링크': '본사링크',
        '기본수량(2)': '고려 기본수량',
        '판매단가(V포함)(2)': '판매단가2(VAT포함)',
        '가격차이(2)': '고려 가격차이',
        '가격차이(2)%': '고려 가격차이(%)',
        '고려기프트 상품링크': '고려 링크',
        '기본수량(3)': '네이버 기본수량',
        '판매단가(V포함)(3)': '판매단가3(VAT포함)',
        '가격차이(3)': '네이버 가격차이',
        '가격차이(3)%': '네이버가격차이(%)',
        '공급사명': '네이버 공급사명',
        '네이버 쇼핑 링크': '네이버 링크',
        '본사 이미지': '해오름(이미지링크)',
        '고려기프트 이미지': '고려기프트(이미지링크)',
        '네이버 이미지': '네이버쇼핑(이미지링크)'
    }
    
    # Rename columns that exist in the DataFrame
    rename_map = {k: v for k, v in column_mapping.items() if k in df_processed.columns}
    df_result = df_processed.rename(columns=rename_map)
    
    # Extract image links from Excel formulas if present
    image_cols = ['해오름(이미지링크)', '고려기프트(이미지링크)', '네이버쇼핑(이미지링크)']
    for col in image_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].astype(str).replace(r'=IMAGE\("(.+)",\s*\d+\)', r'\1', regex=True)
            df_result[col] = df_result[col].replace(['nan', 'None'], '', regex=True)
    
    # Arrange columns in the order specified in the example table
    output_cols = [
        '구분(A/P)', '담당자', '공급사명', '공급처코드', '상품코드', '카테고리(중분류)',
        '상품명', '본사 기본수량', '판매단가1(VAT포함)', '본사링크',
        '고려 기본수량', '판매단가2(VAT포함)', '고려 가격차이', '고려 가격차이(%)', '고려 링크',
        '네이버 기본수량', '판매단가3(VAT포함)', '네이버 가격차이', '네이버가격차이(%)',
        '네이버 공급사명', '네이버 링크',
        '해오름(이미지링크)', '고려기프트(이미지링크)', '네이버쇼핑(이미지링크)'
    ]
    
    # Select only columns that exist in the DataFrame
    final_cols = [col for col in output_cols if col in df_result.columns]
    df_result = df_result[final_cols]
    
    logger.info(f"Final output has {len(df_result)} rows and {len(final_cols)} columns")
    return df_result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        process_second_stage(input_file)
    else:
        print("Usage: python second_stage_processor.py <input_excel_file>") 