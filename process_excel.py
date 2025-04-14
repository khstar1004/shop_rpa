#!/usr/bin/env python3
"""
엑셀 파일 처리 스크립트.

이 스크립트는 ExcelManager 클래스를 사용하여
다양한 엑셀 처리 작업을 수행합니다.

인자에 따라 특정 함수를 호출합니다.
"""

import argparse
import logging
import os
import sys
from urllib.parse import urlparse # For URL check in insert_image
import pandas as pd
import numpy as np
from datetime import datetime
import re
import shutil

# 프로젝트 루트 경로 설정 (스크립트 위치 기반)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# --- Configuration and Manager Setup ---
from utils.config import load_config
from core.processing.excel_manager import ExcelManager

# --- 필요한 함수 임포트 (Openpyxl 관련 기능 직접 사용 for insert_image) ---
from utils.excel_utils import (
    insert_image_to_cell, # Keep for direct use in image insertion command
    download_image, # Keep for direct use in image insertion command
)
from openpyxl import load_workbook # Keep for direct use in image insertion command
from openpyxl.utils.exceptions import InvalidFileException # Keep for direct use in image insertion command
from pathlib import Path # Keep for temporary file handling in image insertion command

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
    parser = argparse.ArgumentParser(description="Process Excel files using ExcelManager.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'check' command -> Uses ExcelManager.check_excel_file
    parser_check = subparsers.add_parser("check", help="Check Excel file structure using ExcelManager.")
    parser_check.add_argument("file_path", help="Path to the Excel file (.xlsx, .xlsm)")

    # 'convert' command -> Uses ExcelManager.convert_xls_to_xlsx
    parser_convert = subparsers.add_parser("convert", help="Convert XLS to XLSX using ExcelManager.")
    parser_convert.add_argument("directory", help="Directory containing the XLS file.")

    # 'format' command -> Uses ExcelManager.add_hyperlinks_to_excel
    # Note: ExcelManager method might apply more formatting than just hyperlinks.
    parser_format = subparsers.add_parser("format", help="Add hyperlinks and format using ExcelManager.")
    parser_format.add_argument("file_path", help="Path to the Excel file (.xlsx, .xlsm)")

    # 'filter' command -> Uses ExcelManager.filter_excel_by_price_diff
    parser_filter = subparsers.add_parser("filter", help="Filter by negative price diff using ExcelManager.")
    parser_filter.add_argument("file_path", help="Path to the Excel file (.xlsx, .xlsm)")

    # 'insert_image' command (uses openpyxl directly, retains excel_utils helpers)
    parser_image = subparsers.add_parser(
        "insert_image", help="Insert image into a cell using openpyxl."
    )
    parser_image.add_argument("excel_file", help="Path to the target Excel file (.xlsx)")
    parser_image.add_argument("image_path", help="Path or URL to the image file")
    parser_image.add_argument("target_cell", help="Target cell address (e.g., 'A1')")
    parser_image.add_argument("--sheet_name", help="Target sheet name (optional, defaults to active sheet)", default=None)

    # 'process_all' command -> Uses ExcelManager.post_process_excel_file
    parser_all = subparsers.add_parser(
        "process_all", help="Run full processing pipeline using ExcelManager's post_process_excel_file."
    )
    parser_all.add_argument("input_path", help="Path to the input Excel file (.xls or .xlsx)")

    args = parser.parse_args()

    # --- Load Config and Instantiate Manager ---
    try:
        config = load_config()
        manager = ExcelManager(config, logger)
    except Exception as e:
        logger.error(f"Failed to load configuration or initialize ExcelManager: {e}", exc_info=True)
        sys.exit(1)

    # --- Command Execution ---
    if args.command == "check":
        logger.info(f"Executing 'check' command for {args.file_path}")
        try:
            # Use ExcelManager's check_excel_file method
            manager.check_excel_file(args.file_path)
            logger.info(f"Check complete for {args.file_path}")
        except Exception as e:
            logger.error(f"Error during check: {e}", exc_info=True)

    elif args.command == "convert":
        logger.info(f"Executing 'convert' command for directory {args.directory}")
        try:
            # Use ExcelManager's convert_xls_to_xlsx method
            converted_path = manager.convert_xls_to_xlsx(args.directory)
            if converted_path:
                logger.info(f"Conversion successful: {converted_path}")
            else:
                logger.warning(f"Conversion did not produce a file or failed.")
        except Exception as e:
            logger.error(f"Error during conversion: {e}", exc_info=True)

    elif args.command == "format":
        logger.info(f"Executing 'format' command for {args.file_path}")
        try:
            # Use ExcelManager's add_hyperlinks_to_excel method
            # Note: This method in ExcelManager might do more than just hyperlinks.
            formatted_path = manager.add_hyperlinks_to_excel(args.file_path)
            if formatted_path != args.file_path:
                 logger.info(f"Formatting successful, output: {formatted_path}")
            else:
                 logger.warning(f"Formatting did not produce a new file or failed. Original file: {args.file_path}")
        except Exception as e:
            logger.error(f"Error during formatting: {e}", exc_info=True)

    elif args.command == "filter":
        logger.info(f"Executing 'filter' command for {args.file_path}")
        try:
            # Use ExcelManager's filter_excel_by_price_diff method
            filtered_path = manager.filter_excel_by_price_diff(args.file_path)
            if filtered_path != args.file_path:
                logger.info(f"Filtering successful, output: {filtered_path}")
            else:
                 logger.warning(f"Filtering did not produce a new file or failed. Original file: {args.file_path}")
        except Exception as e:
            logger.error(f"Error during filtering: {e}", exc_info=True)

    elif args.command == "insert_image":
        logger.info(
            f"Executing 'insert_image' for {args.excel_file}, image {args.image_path}, cell {args.target_cell}"
        )
        # --- Retained image insertion logic using openpyxl directly ---
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
                # Use download_image from excel_utils
                img_bytes = download_image(image_source)
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

            # Insert image using openpyxl function from excel_utils
            insert_image_to_cell(ws, local_image_path, target_cell)

            # Save the workbook
            wb.save(excel_file_path)
            logger.info(f"Image inserted into {target_cell} in {excel_file_path} and file saved.")

        except Exception as e:
            logger.error(f"An error occurred during image insertion: {e}", exc_info=True)
        finally:
            # Clean up temporary file if created
            if temp_file_created and local_image_path and os.path.exists(local_image_path):
                try:
                    # Ensure the temp directory exists before attempting removal
                    temp_dir_path = Path(local_image_path).parent
                    os.remove(local_image_path)
                    logger.info(f"Removed temporary image file: {local_image_path}")
                    # Attempt to remove the directory if it's empty
                    try:
                        temp_dir_path.rmdir()
                        logger.info(f"Removed temporary image directory: {temp_dir_path}")
                    except OSError:
                        # Directory might not be empty, which is fine
                        logger.debug(f"Temporary image directory not empty, not removed: {temp_dir_path}")
                except OSError as e_remove:
                    logger.warning(f"Failed to remove temporary image file {local_image_path}: {e_remove}")
        # --- End of retained logic ---

    elif args.command == "process_all":
        logger.info(f"Executing 'process_all' command for {args.input_path}")
        try:
            # Use ExcelManager's post_process_excel_file method
            # This method should handle conversion, cleaning, formatting, linking etc.
            final_output = manager.post_process_excel_file(args.input_path)

            if final_output and final_output != args.input_path:
                logger.info(f"Processing complete. Final output: {final_output}")
            elif final_output:
                 logger.info(f"Processing complete. No changes made or output is the same as input: {final_output}")
            else:
                logger.error("Processing failed.")
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)

    else:
        parser.print_help()

# --- Removed process_first_to_second_stage function ---
# This logic should reside within ExcelManager or a dedicated reporting/processing module.

if __name__ == "__main__":
    main() 