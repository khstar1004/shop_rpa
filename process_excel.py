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


if __name__ == "__main__":
    main() 