#!/usr/bin/env python3
"""
엑셀 파일 처리 스크립트.

이 스크립트는 utils.excel_utils 모듈의 함수들을 사용하여
다양한 엑셀 처리 작업을 수행합니다.

인자에 따라 특정 함수를 호출합니다.
"""

import argparse
import logging
import os
import sys
from urllib.parse import urlparse # For URL check
import pandas as pd
import numpy as np
from datetime import datetime
import re
import shutil

# 프로젝트 루트 경로 설정 (스크립트 위치 기반)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# --- 필요한 함수 임포트 ---
from utils.excel_utils import (
    check_excel_columns,
    convert_xls_to_xlsx,
    add_hyperlinks_to_excel,
    filter_excel_by_price_diff,
    insert_image_to_cell, # Use openpyxl version
    process_excel_file,
    download_image, # Import download_image for URL handling
)
from openpyxl import load_workbook # Import openpyxl functions
from openpyxl.utils.exceptions import InvalidFileException
from pathlib import Path # For temporary file handling

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("process_excel.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# --- 메인 함수 ---
def main():
    """스크립트 실행 진입점"""
    parser = argparse.ArgumentParser(description="Process Excel files.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'check' command
    parser_check = subparsers.add_parser("check", help="Check and add columns to Excel.")
    parser_check.add_argument("file_path", help="Path to the Excel file (.xlsx)")

    # 'convert' command
    parser_convert = subparsers.add_parser("convert", help="Convert XLS to XLSX.")
    # Changed argument to directory
    parser_convert.add_argument("directory", help="Directory containing XLS files")

    # 'format' command
    parser_format = subparsers.add_parser("format", help="Add hyperlinks and format.")
    parser_format.add_argument("file_path", help="Path to the Excel file (.xlsx)")

    # 'filter' command
    parser_filter = subparsers.add_parser("filter", help="Filter by negative price diff.")
    parser_filter.add_argument("file_path", help="Path to the Excel file (.xlsx)")

    # 'insert_image' command (modified)
    parser_image = subparsers.add_parser(
        "insert_image", help="Insert image into a cell using openpyxl."
    )
    parser_image.add_argument("excel_file", help="Path to the target Excel file (.xlsx)")
    parser_image.add_argument("image_path", help="Path or URL to the image file")
    parser_image.add_argument("target_cell", help="Target cell address (e.g., 'A1')")
    parser_image.add_argument("--sheet_name", help="Target sheet name (optional, defaults to active sheet)", default=None)

    # 'process_all' command
    parser_all = subparsers.add_parser(
        "process_all", help="Run full processing pipeline (convert, check, filter, format)."
    )
    parser_all.add_argument("input_path", help="Path to the input Excel file (.xls or .xlsx)")

    args = parser.parse_args()

    # --- Command Execution ---
    if args.command == "check":
        logger.info(f"Executing 'check' command for {args.file_path}")
        check_excel_columns(args.file_path)
    elif args.command == "convert":
        logger.info(f"Executing 'convert' command for directory {args.directory}")
        convert_xls_to_xlsx(args.directory)
    elif args.command == "format":
        logger.info(f"Executing 'format' command for {args.file_path}")
        add_hyperlinks_to_excel(args.file_path)
    elif args.command == "filter":
        logger.info(f"Executing 'filter' command for {args.file_path}")
        filter_excel_by_price_diff(args.file_path)

    elif args.command == "insert_image":
        logger.info(
            f"Executing 'insert_image' for {args.excel_file}, image {args.image_path}, cell {args.target_cell}"
        )
        # --- Modified image insertion logic --- 
        excel_file_path = args.excel_file
        image_source = args.image_path
        target_cell = args.target_cell
        sheet_name = args.sheet_name
        local_image_path = None
        temp_file_created = False

        try:
            # Handle URL images: Download first
            if urlparse(image_source).scheme in ["http", "https"]:
                logger.info(f"Downloading image from URL: {image_source}")
                img_bytes = download_image(image_source) # Use the existing download function
                if img_bytes:
                    # Create a temporary file path
                    temp_dir = Path(os.path.dirname(excel_file_path)) / "temp_images"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    # Generate a safe filename (replace invalid chars)
                    safe_filename = "".join(c if c.isalnum() or c in ('.', '_') else '_' for c in Path(image_source).name)
                    if not safe_filename: safe_filename = f"temp_image_{os.urandom(4).hex()}.png"
                    temp_file_path = temp_dir / safe_filename
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(img_bytes)
                    local_image_path = str(temp_file_path)
                    temp_file_created = True
                    logger.info(f"Image downloaded and saved to temporary path: {local_image_path}")
                else:
                    logger.error(f"Failed to download image from URL: {image_source}")
                    return # Exit if download fails
            else:
                # Assume local path
                local_image_path = image_source

            # Check if local image path exists
            if not local_image_path or not os.path.exists(local_image_path):
                logger.error(f"Local image file not found or could not be obtained: {local_image_path}")
                return

            # Load workbook and get worksheet
            try:
                wb = load_workbook(excel_file_path)
                if sheet_name:
                    ws = wb[sheet_name]
                else:
                    ws = wb.active # Default to active sheet
            except FileNotFoundError:
                 logger.error(f"Excel file not found: {excel_file_path}")
                 return
            except KeyError:
                 logger.error(f"Sheet '{sheet_name}' not found in {excel_file_path}")
                 return
            except InvalidFileException:
                 logger.error(f"Invalid Excel file (maybe corrupted or wrong format): {excel_file_path}")
                 return

            # Insert image using openpyxl function
            insert_image_to_cell(ws, local_image_path, target_cell)
            # Note: insert_image_to_cell handles its own logging/errors

            # Save the workbook
            wb.save(excel_file_path)
            logger.info(f"Image inserted into {target_cell} in {excel_file_path} and file saved.")

        except Exception as e:
            logger.error(f"An error occurred during image insertion: {e}", exc_info=True)
        finally:
            # Clean up temporary file if created
            if temp_file_created and local_image_path and os.path.exists(local_image_path):
                try:
                    os.remove(local_image_path)
                    logger.info(f"Removed temporary image file: {local_image_path}")
                except OSError as e_remove:
                    logger.warning(f"Failed to remove temporary image file {local_image_path}: {e_remove}")
        # --- End of modified logic ---

    elif args.command == "process_all":
        logger.info(f"Executing 'process_all' command for {args.input_path}")
        final_output = process_excel_file(args.input_path)
        if final_output:
            logger.info(f"Processing complete. Final output: {final_output}")
        else:
            logger.error("Processing failed.")
    else:
        parser.print_help()

def process_first_to_second_stage(input_file, output_dir=None):
    """
    작업메뉴얼에 따라 1차 파일에서 2차 파일 생성
    
    규칙:
    1. 노란색 셀(가격차이 음수) 상품만 2차 파일로 이동
    2. 네이버쇼핑 [기본수량] 없는 상품 중 가격차이(%) ≤ 10% -> 삭제
    3. 가격차이 양수(+) 상품 -> 삭제
    4. 고려기프트/네이버쇼핑에 가격불량 기록 전무 -> 줄 삭제
    5. 이미지 제거, 링크만 남김
    
    Args:
        input_file: 1차 파일 경로
        output_dir: 출력 디렉토리 (없으면 입력 파일과 동일 디렉토리)
        
    Returns:
        str: 생성된 2차 파일 경로
    """
    try:
        logger.info(f"1차 -> 2차 파일 변환 시작: {input_file}")
        
        # 입력 파일 유효성 검사
        if not input_file or not os.path.exists(input_file):
            logger.error(f"입력 파일이 존재하지 않습니다: {input_file}")
            raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_file}")
            
        # 파일 확장자 확인
        _, ext = os.path.splitext(input_file)
        if ext.lower() not in ['.xls', '.xlsx', '.xlsm', '.csv']:
            logger.error(f"지원되지 않는 파일 형식입니다: {ext}")
            raise ValueError(f"지원되지 않는 파일 형식입니다: {ext}")
        
        # 출력 디렉토리 설정
        if not output_dir:
            output_dir = os.path.dirname(input_file)
        
        # 출력 디렉토리 존재 확인 및 생성    
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"출력 디렉토리 생성: {output_dir}")
            except Exception as e:
                logger.error(f"출력 디렉토리 생성 실패: {str(e)}")
                raise
            
        # 출력 파일명 생성 (파일명-result.xlsx)
        input_filename = os.path.basename(input_file)
        file_base, file_ext = os.path.splitext(input_filename)
        output_filename = f"{file_base}-result.xlsx"  # 출력은 항상 xlsx 형식으로 통일
        output_file = os.path.join(output_dir, output_filename)
        
        # 1차 파일 로드
        try:
            # CSV 파일인 경우 특별 처리
            if ext.lower() == '.csv':
                df = pd.read_csv(input_file, encoding='utf-8')
                logger.info(f"CSV 파일 로드 완료: {len(df)}행")
            else:
                df = pd.read_excel(input_file)
                logger.info(f"Excel 파일 로드 완료: {len(df)}행")
        except Exception as e:
            logger.error(f"입력 파일 로드 중 오류: {str(e)}")
            raise
        
        # 필수 컬럼 확인
        required_columns = ['가격차이(2)', '가격차이(3)', '기본수량(3)', '가격차이(3)%']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"필수 컬럼이 누락되었습니다: {missing_columns}")
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        # 원본 데이터 백업
        df_original = df.copy()
        
        # 구분값(A/P) 추출 및 보존
        has_distinction = '구분' in df.columns
        distinction_values = {}
        if has_distinction:
            for idx, row in df.iterrows():
                if pd.notna(row.get('상품Code')):
                    distinction_values[str(row.get('상품Code'))] = row.get('구분', 'A')
        
        # 데이터 유형 변환 (숫자형 컬럼)
        numeric_columns = ['가격차이(2)', '가격차이(3)', '가격차이(3)%']
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)  # NaN을 0으로 대체
            except Exception as e:
                logger.warning(f"컬럼 '{col}' 숫자 변환 중 오류: {str(e)}")
        
        # 1. 노란색 셀(가격차이 음수) 상품 필터링
        # 가격차이(2) 또는 가격차이(3)가 음수인 행만 선택
        try:
            df_filtered = df[
                (df['가격차이(2)'] < 0) | 
                (df['가격차이(3)'] < 0)
            ].copy()
            
            logger.info(f"규칙 1 적용 후: {len(df_filtered)}행 (음수 가격차이)")
        except Exception as e:
            logger.error(f"규칙 1 적용 중 오류: {str(e)}")
            # 오류 발생 시도 계속 진행
            df_filtered = df.copy()
        
        # 2. 네이버쇼핑 [기본수량] 없는 상품 중 가격차이(%) ≤ 10% -> 삭제
        # 기본수량(3)이 비어있고 가격차이(3)%가 -10% 이상인 행 제거
        try:
            # 먼저 값이 없거나 NaN인 행 대응
            has_basic_quantity = df_filtered['기본수량(3)'].notna() & (df_filtered['기본수량(3)'] != '')
            # 값이 있는 행 중 숫자로 변환 가능한지 확인
            numeric_basic_quantity = pd.to_numeric(df_filtered.loc[has_basic_quantity, '기본수량(3)'], errors='coerce')
            
            # 필터링을 위한 조건 준비
            missing_basic_quantity = ~has_basic_quantity | numeric_basic_quantity.isna()
            
            # 가격차이(3)% 변환
            price_diff_percent = pd.to_numeric(df_filtered['가격차이(3)%'], errors='coerce')
            price_diff_percent = price_diff_percent.fillna(0)  # NaN을 0으로 대체
            small_price_diff = (price_diff_percent >= -10)
            
            # 조건: 기본수량 없고 가격차이 작은(-10% 이상) 행 제외
            rows_to_delete = missing_basic_quantity & small_price_diff
            df_filtered = df_filtered.loc[~rows_to_delete].copy()
            
            logger.info(f"규칙 2 적용 후: {len(df_filtered)}행 (기본수량 없는 작은 가격차이 제거)")
        except Exception as e:
            logger.error(f"규칙 2 적용 중 오류: {str(e)}")
            # 오류 시 이 규칙은 건너뜀
        
        # 3. 가격차이 양수(+) 상품 -> 삭제
        try:
            df_filtered = df_filtered[
                ~((df_filtered['가격차이(2)'] > 0) & 
                  (df_filtered['가격차이(3)'] > 0))
            ].copy()
            
            logger.info(f"규칙 3 적용 후: {len(df_filtered)}행 (양수 가격차이 제거)")
        except Exception as e:
            logger.error(f"규칙 3 적용 중 오류: {str(e)}")
            # 오류 시 이 규칙은 건너뜀
        
        # 4. 고려기프트/네이버쇼핑에 가격불량 기록 전무 -> 줄 삭제
        try:
            # 양쪽 모두 가격불량(음수 가격차이)이 없는 행 제거
            has_koryo_price_issue = (df_filtered['가격차이(2)'] < 0)
            has_naver_price_issue = (df_filtered['가격차이(3)'] < 0)
            
            df_filtered = df_filtered[
                has_koryo_price_issue | has_naver_price_issue
            ].copy()
            
            logger.info(f"규칙 4 적용 후: {len(df_filtered)}행 (가격불량 없는 행 제거)")
        except Exception as e:
            logger.error(f"규칙 4 적용 중 오류: {str(e)}")
            # 오류 시 이 규칙은 건너뜀
        
        # 결과가 비어있으면 기록
        if len(df_filtered) == 0:
            logger.warning("필터링 후 데이터가 0행입니다. 조건에 맞는 행이 없을 수 있습니다.")
        
        # 5. 이미지 관련 처리 - 이미지 URL 컬럼만 유지하고 실제 이미지는 제거
        # 이미지 파일이 있는 열인지 확인하고 처리
        image_columns = [col for col in df_filtered.columns if '이미지' in col.lower()]
        logger.info(f"이미지 관련 컬럼: {image_columns}")
        
        # 구분값(A/P) 복원 - 작업메뉴얼 요구사항
        if has_distinction and '상품Code' in df_filtered.columns and '구분' in df_filtered.columns:
            for idx, row in df_filtered.iterrows():
                product_code = str(row.get('상품Code', ''))
                if product_code in distinction_values:
                    df_filtered.at[idx, '구분'] = distinction_values[product_code]
            logger.info("구분값(A/P) 복원 완료")
        
        # 결과 저장
        try:
            df_filtered.to_excel(output_file, index=False)
            logger.info(f"2차 파일 생성 완료: {output_file} (최종 {len(df_filtered)}행)")
            return output_file
        except Exception as e:
            logger.error(f"결과 파일 저장 중 오류: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"1차 -> 2차 파일 변환 중 오류: {str(e)}")
        raise

if __name__ == "__main__":
    main() 