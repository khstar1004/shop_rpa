"""
Excel 파일 처리를 위한 유틸리티 함수들

PythonScript 폴더의 Excel 관련 기능을 활용하여 일반적인 Excel 작업을 쉽게 수행할 수 있는
유틸리티 함수들을 제공합니다.
"""

import logging
import os
import re
import zipfile # Added for exception handling
from datetime import datetime
from typing import Optional, Any
from urllib.parse import urlparse # Moved import
import io

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, PatternFill, Side, Font, NamedStyle
from openpyxl.cell import Cell
from openpyxl.drawing.image import Image as OpenpyxlImage # Renamed to avoid conflict
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import IllegalCharacterError
from openpyxl.worksheet.worksheet import Worksheet
from PIL import Image as PILImage # Added Pillow import
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    IMAGE = None

# 기본 로거 설정
logger = logging.getLogger(__name__)

# --- Formatting and Styling Functions ---

DEFAULT_FONT = Font(name="맑은 고딕", size=10)
HEADER_FONT = Font(name="맑은 고딕", size=10, bold=True)
CENTER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT_ALIGNMENT = Alignment(horizontal="left", vertical="center", wrap_text=True)
THIN_BORDER_SIDE = Side(style="thin")
THIN_BORDER = Border(
    left=THIN_BORDER_SIDE,
    right=THIN_BORDER_SIDE,
    top=THIN_BORDER_SIDE,
    bottom=THIN_BORDER_SIDE,
)
GRAY_FILL = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
YELLOW_FILL = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
LIGHT_YELLOW_FILL = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")
LIGHT_BLUE_FILL = PatternFill(start_color="CCFFFF", end_color="CCFFFF", fill_type="solid")


# Define NamedStyles
header_style = NamedStyle(
    name="header_style",
    font=HEADER_FONT,
    border=THIN_BORDER,
    alignment=CENTER_ALIGNMENT,
    fill=GRAY_FILL,
)
default_style = NamedStyle(
    name="default_style", font=DEFAULT_FONT, border=THIN_BORDER, alignment=LEFT_ALIGNMENT
)
center_style = NamedStyle(
    name="center_style",
    font=DEFAULT_FONT,
    border=THIN_BORDER,
    alignment=CENTER_ALIGNMENT,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_image(url: str, timeout: int = 10) -> Optional[bytes]:
    """Downloads an image from a URL."""
    # (Keep existing download_image function)
    # ... existing code ...


def insert_image_to_cell(ws: Worksheet, image_path: str, cell_address: str, size: tuple = (100, 100)):
    """
    Inserts a resized image into the specified cell using openpyxl.

    Args:
        ws: The openpyxl worksheet object.
        image_path: Path to the local image file.
        cell_address: The target cell (e.g., 'A1').
        size: Tuple of desired (width, height) in pixels.
    """
    try:
        if not os.path.exists(image_path):
            logging.warning(f"Image path does not exist: {image_path}")
            return

        # Resize image using Pillow
        img_pil = PILImage.open(image_path)
        img_pil = img_pil.convert("RGB")  # Ensure compatibility, remove alpha
        img_pil.thumbnail(size) # Resize while maintaining aspect ratio

        # Prepare image for openpyxl
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG') # Save resized image to byte stream
        img_byte_arr.seek(0)

        img_openpyxl = OpenpyxlImage(img_byte_arr)

        # Set image dimensions (optional, as thumbnail maintains aspect ratio)
        # img_openpyxl.width, img_openpyxl.height = size

        # Add image to worksheet anchored to the cell
        ws.add_image(img_openpyxl, cell_address)

        # Adjust row height and column width (approximations based on Excelsaveimage.py)
        target_cell = ws[cell_address]
        # Excel row height is in points (1 point = 1/72 inch). Pillow size is in pixels.
        # Approximation: 1 pixel ~ 0.75 points (at 96 DPI)
        row_height_points = size[1] * 0.75
        ws.row_dimensions[target_cell.row].height = max(ws.row_dimensions[target_cell.row].height or 0, row_height_points)

        # Excel column width is in characters. Very approximate conversion.
        # Using the ratio from Excelsaveimage.py: width_pixels / 6.25
        col_width_chars = size[0] / 7.0 # Adjusted divisor slightly based on common observations
        col_letter = get_column_letter(target_cell.column)
        ws.column_dimensions[col_letter].width = max(ws.column_dimensions[col_letter].width or 0, col_width_chars)

        logging.debug(f"Inserted image {os.path.basename(image_path)} into cell {cell_address}, adjusted row/col size.")

    except FileNotFoundError:
        logging.warning(f"Image file not found: {image_path}")
    except Exception as e:
        logging.error(f"Failed to insert image {image_path} into cell {cell_address}: {e}", exc_info=True)


def clean_value(value: Any) -> Any:
    """Cleans value for Excel insertion."""
    # (Keep existing clean_value function)
    # ... existing code ...


def check_excel_columns(file_path: str) -> bool:
    """
    엑셀 파일에 필요한 컬럼이 있는지 확인하고 없으면 추가합니다.

    Args:
        file_path: 엑셀 파일 경로

    Returns:
        bool: 성공 여부
    """
    try:
        logger.info("Checking Excel file: %s", file_path)

        # 파일 존재 확인 (FileNotFoundError 는 아래 read_excel 에서 처리될 수 있음)
        if not os.path.exists(file_path):
            logger.error("Excel file not found at path: %s", file_path)
            return False

        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)

        # 추가할 컬럼 정의
        columns_to_add = ["본사 이미지", "고려기프트 이미지", "네이버 이미지"]
        need_to_modify = False

        # 컬럼 확인 및 추가
        for column in columns_to_add:
            if column not in df.columns:
                df[column] = ""
                need_to_modify = True
                logger.info("Added missing column: %s", column)

        if not need_to_modify:
            logger.info("All required columns exist. No modifications needed.")
            return True

        # 컬럼명의 앞뒤 공백 제거
        df.columns = [col.strip() for col in df.columns]

        # 엑셀 파일 저장 (원본 덮어쓰기)
        df.to_excel(file_path, index=False)
        logger.info("Updated Excel file with required columns: %s", file_path)

        return True

    except FileNotFoundError:
        logger.error("Excel file not found during read/write: %s", file_path)
        return False
    except (pd.errors.ParserError, ValueError) as e:
        logger.error("Error parsing Excel file: %s", e)
        return False
    except (OSError, IOError) as e:
        logger.error("File system error checking/writing Excel file: %s", e)
        return False
    # Use specific exception types instead of Exception
    except (AttributeError, KeyError, TypeError) as e:
        logger.error("Data processing error checking Excel columns: %s", e)
        return False


def convert_xls_to_xlsx(input_directory: str) -> str:
    """
    XLS 파일을 XLSX 형식으로 변환합니다.

    Args:
        input_directory: 입력 디렉토리 경로

    Returns:
        str: 변환된 XLSX 파일 경로 또는 빈 문자열
    """
    output_file_path = ""
    file_name = "" # Initialize file_name
    try:
        logger.info("Looking for XLS files in: %s", input_directory)

        # XLS 파일 찾기
        xls_files = [
            f for f in os.listdir(input_directory) if f.lower().endswith(".xls")
        ]

        if not xls_files:
            logger.warning("No XLS files found in the directory.")
            return ""

        # 첫 번째 XLS 파일 처리
        file_name = xls_files[0]
        file_path = os.path.join(input_directory, file_name)

        logger.info("Converting XLS file: %s", file_path)

        # XLS 파일 로드 (pd.read_html 사용)
        try:
            tables = pd.read_html(file_path, encoding="cp949")
            if not tables:
                logger.error("No tables found in XLS file: %s", file_path)
                return ""
            df = tables[0]
        except ImportError:
             logger.error("Optional dependency 'lxml' or 'html5lib' not found for pd.read_html.")
             return ""
        except ValueError as e:
            # pd.read_html can raise ValueError for various parsing issues
            logger.error("Error parsing XLS file %s with read_html: %s", file_path, e)
            return ""
        # Use specific exception types instead of Exception
        except (OSError, IOError) as e:
            logger.error("File error reading XLS file %s with read_html: %s", file_path, str(e))
            return ""

        # 첫 번째 행을 헤더로 사용
        if df.empty:
            logger.warning("XLS file %s seems empty after reading.", file_path)
            return ""
        df.columns = df.iloc[0].astype(str).str.strip() # Ensure header is string
        df = df.drop(0).reset_index(drop=True)

        # 상품명 전처리 (// 이후 제거 및 특수 문자 정리)
        # Corrected indentation
        pattern = r"(\d{4}_[A-Z]\.)|(\d+\+\d+)|[^a-zA-Z0-9가-힣\s]|\s+"

        def preprocess_product_name(product_name):
            """상품명에서 불필요한 부분(주석, 특정 키워드, 특수문자 등)을 제거합니다."""
            # Corrected indentation
            if not isinstance(product_name, str):
                 return str(product_name) # Convert non-strings

            # // 이후 제거
            # Corrected indentation
            if "//" in product_name:
                 product_name = product_name.split("//")[0]

            # 특수 문자 및 패턴 제거
            product_name = re.sub(pattern, " ", product_name)
            # Wrapped long line
            product_name = (
                product_name.replace("정품", "")
                .replace("NEW", "")
                .replace("특가", "")
            )
            # Corrected indentation and wrapped long lines
            product_name = product_name.replace("주문제작타올", "").replace(
                 "주문제작수건", ""
            )
            product_name = product_name.replace("결혼답례품 수건", "").replace(
                 "답례품수건", ""
            )
            product_name = product_name.replace("주문제작 수건", "").replace(
                 "돌답례품수건", ""
            )
            # Corrected indentation and wrapped long lines
            product_name = (
                 product_name.replace("명절선물세트", "")
                 .replace("각종행사수건", "")
                 .strip()
            )
            # Corrected indentation
            product_name = re.sub(" +", " ", product_name)
            return product_name

        if "상품명" in df.columns:
            df["상품명"] = df["상품명"].apply(preprocess_product_name)

        # 필요한 컬럼 추가
        for column in ["본사 이미지", "고려기프트 이미지", "네이버 이미지"]:
            if column not in df.columns:
                df[column] = ""

        # 출력 파일명 설정
        output_file_name = file_name.replace(".xls", ".xlsx")
        output_file_path = os.path.join(input_directory, output_file_name)

        # XLSX로 저장
        df.to_excel(output_file_path, index=False)
        logger.info("Converted file saved to: %s", output_file_path)

        return output_file_path

    except FileNotFoundError:
        # Corrected indentation
        logger.error("Directory or file not found during XLS conversion: %s / %s", input_directory, file_name)
        return ""
    except (OSError, IOError) as e:
        # Corrected indentation
        logger.error("File system error during XLS conversion: %s", e)
        # Clean up potentially created output file
        if output_file_path and os.path.exists(output_file_path):
             try:
                 os.remove(output_file_path)
             except OSError as rm_err:
                 logger.error("Failed to remove partial XLSX file: %s", rm_err)
        return ""
    except (AttributeError, TypeError, KeyError, IndexError) as e:
         # Corrected indentation
         logger.error("Data processing error during XLS conversion (check DataFrame structure): %s", e)
         return ""
    # Keep Exception for truly unexpected issues, but log traceback if possible
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error("Unexpected error converting XLS to XLSX: %s", str(e), exc_info=True)
        # Clean up potentially created output file
        if output_file_path and os.path.exists(output_file_path):
             # Corrected indentation
             try:
                 os.remove(output_file_path)
             except OSError as rm_err:
                 # Corrected indentation
                 logger.error("Failed to remove partial XLSX file: %s", rm_err)
        return ""


def add_hyperlinks_to_excel(file_path: str) -> str:
    """
    엑셀 파일의 URL 필드를 하이퍼링크로 변환하고 기본 서식을 적용합니다.

    입력 파일은 수정하지 않고, "_formatted_{timestamp}.xlsx" 접미사가 붙은
    새로운 파일에 결과를 저장합니다. 서식에는 테두리, 자동 줄 바꿈, 헤더 강조,
    가격 차이 음수 행 노란색 배경 적용이 포함됩니다.

    Args:
        file_path: 입력 엑셀 파일 경로 (.xlsx, .xls, .xlsm)

    Returns:
        str: 처리된 새 파일 경로. 오류 발생 시 원본 파일 경로를 반환합니다.
    """
    output_file_path = file_path # Default return on error
    try:
        logger.info("Adding hyperlinks and formatting to Excel file: %s", file_path)

        # 출력 파일명 생성 (기존 파일 덮어쓰지 않음)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Wrapped long line
        output_filename = f"{base_name}_formatted_{timestamp}.xlsx"
        output_file_path = os.path.join(output_directory, output_filename)

        # DataFrame 로드
        df = pd.read_excel(file_path)

        # 워크북 로드 (openpyxl 사용)
        try:
             wb = load_workbook(file_path)
             ws = wb.active
        except FileNotFoundError: # Already specific
             logger.error("Input file not found for openpyxl: %s", file_path)
             return file_path
        # Catch more specific openpyxl/zipfile load errors
        except (zipfile.BadZipFile, KeyError, ValueError, TypeError) as e:
             logger.error("Error loading workbook with openpyxl: %s", e)
             return file_path

        # 가격 비교 컬럼 찾기
        price_diff_col_letter = None
        for col in ws.iter_cols(min_row=1, max_row=1):
            # Corrected indentation
            if col[0].value == "가격차이":
                 price_diff_col_letter = col[0].column_letter
                 break

        # 가격차이 컬럼과 URL 컬럼에 대해 서식 적용
        # Corrected indentation
        url_columns = ["본사 URL", "고려기프트 URL", "네이버 URL"]
        url_col_indices = {}
        # Corrected indentation
        for idx, col_name in enumerate(df.columns):
             if col_name in url_columns:
                  url_col_indices[col_name] = idx + 1 # 1-based index for openpyxl

        # Corrected indentation
        header_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
        # Corrected indentation
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        # Corrected indentation
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        # Corrected indentation
        for row_idx, row in enumerate(ws.iter_rows(min_row=1), start=1):
             is_header = row_idx == 1
             for cell_idx, cell in enumerate(row, start=1):
                  # Corrected indentation
                  cell.border = thin_border
                  cell.alignment = Alignment(wrap_text=True, vertical="center")

                  # Corrected indentation
                  if is_header:
                       cell.fill = header_fill
                  else:
                       # 하이퍼링크 적용 (URL 컬럼 확인)
                       # Corrected indentation
                       col_name = df.columns[cell_idx - 1]
                       # Wrapped long line
                       if col_name in url_col_indices and isinstance(cell.value, str) and cell.value.startswith("http"):
                            # Corrected indentation
                            try:
                                 # Validate URL before setting hyperlink
                                 # Corrected indentation and line wrapping
                                 parsed_url = urlparse(cell.value)
                                 if parsed_url.scheme and parsed_url.netloc:
                                      cell.hyperlink = cell.value
                                      cell.style = "Hyperlink"
                                 else:
                                      # Corrected indentation
                                      logger.warning("Invalid URL format in cell %s%d: %s",
                                                       cell.column_letter, cell.row, cell.value)
                            except ValueError as url_err:
                                 # Corrected indentation
                                 logger.warning("Error parsing URL in cell %s%d: %s (%s)",
                                                  cell.column_letter, cell.row, cell.value, url_err)

                       # 가격 차이 음수 하이라이트
                       # Corrected indentation
                       if price_diff_col_letter and cell.column_letter == price_diff_col_letter:
                            # Corrected indentation
                            try:
                                 if isinstance(cell.value, (int, float)) and cell.value < 0:
                                      cell.fill = yellow_fill
                            except TypeError:
                                 # Corrected indentation
                                 pass # Ignore non-numeric types

        # 열 너비 자동 조정 (근사치)
        # Corrected indentation
        for col in ws.columns:
             max_length = 0
             column = col[0].column_letter # Get the column name
             # Corrected indentation
             for cell in col:
                  try: # Added try-except for potential attribute errors
                       # Corrected indentation
                       if cell.value:
                            # Corrected indentation
                            cell_len = len(str(cell.value))
                            # R1731: Use max() builtin
                            max_length = max(max_length, cell_len)
                  except AttributeError:
                       # Corrected indentation
                       pass # Skip if cell has no value attribute
             # Corrected indentation
             adjusted_width = (max_length + 2) * 1.2
             # Wrapped long line and corrected indentation
             ws.column_dimensions[column].width = min(adjusted_width, 50) # Max width 50

        # 워크북 저장
        wb.save(output_file_path)
        logger.info("Formatted Excel file saved to: %s", output_file_path)
        return output_file_path

    # Catch specific file I/O or saving errors
    except (OSError, IOError, ValueError, TypeError, AttributeError) as e:
        # Corrected indentation
        logger.error("Error processing or saving formatted Excel file: %s", e)
        return file_path # Return original path on error


def filter_excel_by_price_diff(file_path: str) -> str:
    """
    엑셀 파일에서 '가격차이' 열을 기준으로 음수 값을 가진 행을 제외하고,
    결과를 새 파일에 저장합니다.

    입력 파일은 수정하지 않고, "_filtered_{timestamp}.xlsx" 접미사가 붙은
    새로운 파일에 결과를 저장합니다. '가격차이' 열이 없거나 숫자형이 아닌 경우,
    필터링 없이 원본 내용 그대로 새 파일에 저장하고 경고를 로깅합니다.

    Args:
        file_path: 입력 엑셀 파일 경로 (.xlsx)

    Returns:
        str: 필터링된 새 파일 경로. 오류 발생 시 원본 파일 경로를 반환합니다.
    """
    output_file_path = file_path  # Default return on error
    try:
        logger.info("Filtering Excel file by price difference: %s", file_path)

        # 출력 파일명 생성
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Wrapped long line
        output_filename = f"{base_name}_filtered_{timestamp}.xlsx"
        output_file_path = os.path.join(output_directory, output_filename)

        # DataFrame 로드
        df = pd.read_excel(file_path)

        # '가격차이' 컬럼 확인 및 필터링
        price_diff_col = "가격차이"
        if price_diff_col in df.columns:
            try:
                # 숫자형으로 변환 시도, 변환 불가 값은 NaT/NaN으로 처리
                df[price_diff_col] = pd.to_numeric(df[price_diff_col], errors="coerce")
                # 음수가 아닌 행만 유지 (NaN 포함)
                filtered_df = df[df[price_diff_col] >= 0].copy()
                # apply filter rules after filtering rows
                # Note: Applying filter rules to the saved file, not the DataFrame here
                logger.info("Filtered %d rows with negative price difference.", len(df) - len(filtered_df))

            except (ValueError, TypeError):
                # Removed unused variable data_err
                logger.warning(
                    "Could not convert '%s' column to numeric for filtering. Saving unfiltered data.",
                    price_diff_col,
                )
                filtered_df = df.copy() # Use original data if conversion fails
        else:
            logger.warning("Column '%s' not found. Saving unfiltered data.", price_diff_col)
            filtered_df = df.copy() # Use original data if column doesn't exist

        # 필터링된 DataFrame을 새 파일로 저장
        # Corrected indentation
        filtered_df.to_excel(output_file_path, index=False)

        # 저장된 파일에 openpyxl 필터 규칙 적용 (옵션)
        try:
            wb = load_workbook(output_file_path)
            ws = wb.active
            _apply_filter_rules(ws) # Apply filtering rules like autofilter
            wb.save(output_file_path)
        except (FileNotFoundError, zipfile.BadZipFile, KeyError, ValueError, TypeError) as e:
            # W0311: Correct indentation
            # C0301: Wrapped long log message
            logger.warning("Could not apply openpyxl filter rules to '%s': %s",
                         output_file_path, e)
        # W0718: Catch specific exception type instead of Exception
        except AttributeError as e:
            # W0311: Correct indentation
            # C0301: Wrapped long log message
            logger.warning("Unexpected error applying openpyxl filter rules to '%s': %s",
                         output_file_path, e, exc_info=True)

        logger.info("Filtered Excel file saved to: %s", output_file_path)
        return output_file_path

    # Catch specific file I/O or saving errors
    except (OSError, IOError, ValueError, KeyError, TypeError, AttributeError) as e:
        # Corrected indentation
        logger.error("Error filtering or saving Excel file: %s", e)
        return file_path # Return original path on error


def _apply_filter_rules(worksheet) -> None:
    """
    주어진 워크시트에 필터 규칙을 적용합니다.
    (현재는 자동 필터만 적용)

    Args:
        worksheet: 필터를 적용할 openpyxl 워크시트 객체
    """
    try:
        # 워크시트의 사용된 영역에 자동 필터 적용
        # Corrected indentation
        worksheet.auto_filter.ref = worksheet.dimensions
        # 필요하다면 특정 열에 필터 기준 추가 가능
        # 예: worksheet.auto_filter.add_filter_column(0, ["Criteria1", "Criteria2"])
        # 예: worksheet.auto_filter.add_sort_condition("B2:B" + str(worksheet.max_row))

        # '가격차이' 열이 존재하면 0 이상인 값만 표시하도록 필터 추가 시도
        price_diff_col_idx = None
        # Corrected indentation
        for idx, cell in enumerate(worksheet[1]): # Check header row
            if cell.value == "가격차이":
                 price_diff_col_idx = idx
                 break

        if price_diff_col_idx is not None:
             # Implementing a simple '>=' filter is complex with openpyxl's standard filters.
             # Using auto_filter ref covers the basic filter toggle.
             # For value-based filtering, consider filtering pandas DataFrame *before* saving,
             # or using conditional formatting instead of filters.
             logger.debug("Auto filter applied. Manual filtering on '가격차이' might be needed in Excel.")
        else:
             logger.debug("Auto filter applied. '가격차이' column not found for specific rule.")

        logger.info("Applied auto filter rules to the worksheet.")
    # Catch exceptions during openpyxl operations
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning("Could not apply filter rules: %s", e)


def process_excel_file(input_path: str) -> Optional[str]:
    """
    주어진 Excel 파일에 대해 일련의 처리 단계를 수행합니다.

    1. XLS -> XLSX 변환 (필요시)
    2. 필수 컬럼 확인 및 추가
    3. URL 하이퍼링크 변환 및 기본 서식 적용
    4. 가격 차이 기준으로 행 필터링 (옵션)

    Args:
        input_path: 처리할 Excel 파일 경로

    Returns:
        Optional[str]: 최종 처리된 파일 경로. 오류 발생 시 None 반환.
    """
    processed_path = input_path
    try:
        logger.info("Starting processing for Excel file: %s", input_path)

        # 1. XLS -> XLSX 변환
        if input_path.lower().endswith(".xls"):
            logger.info("Input is an XLS file, attempting conversion to XLSX.")
            # Corrected indentation
            converted_path = convert_xls_to_xlsx(os.path.dirname(input_path))
            if not converted_path:
                 logger.error("Failed to convert XLS to XLSX. Aborting processing.")
                 return None
            logger.info("Successfully converted XLS to XLSX: %s", converted_path)
            processed_path = converted_path # Update path after conversion
        elif not input_path.lower().endswith((".xlsx", ".xlsm")):
             logger.error("Input file is not a supported Excel format (.xlsx, .xlsm, .xls): %s", input_path)
             return None

        # 파일 존재 재확인 (변환 후)
        if not os.path.exists(processed_path):
             logger.error("Processed file path does not exist: %s", processed_path)
             return None

        # 2. 필수 컬럼 확인 및 추가
        if not check_excel_columns(processed_path):
            # Corrected indentation
            logger.error("Failed to verify or add required columns. Aborting processing.")
            return None

        # 3. 하이퍼링크 추가 및 서식 적용
        # Corrected indentation
        formatted_path = add_hyperlinks_to_excel(processed_path)
        if formatted_path == processed_path:
             # Corrected indentation
             logger.warning("Hyperlink/formatting step failed or produced no changes.")
             # Continue with the current path if formatting failed but original exists
             if not os.path.exists(processed_path):
                  logger.error("Original processed file disappeared after formatting attempt.")
                  return None
             # No change needed: processed_path remains the same
        else:
             # Corrected indentation
             logger.info("Successfully added hyperlinks and formatting: %s", formatted_path)
             processed_path = formatted_path # Update path after formatting

        # 4. 가격 차이 기준으로 행 필터링 (예: 0 이상만 남기기)
        # 이 단계는 필요에 따라 활성화/비활성화 할 수 있습니다.
        apply_price_filter = True # 설정 또는 조건에 따라 변경 가능
        if apply_price_filter:
            logger.info("Applying price difference filter.")
            # Corrected indentation
            filtered_path = filter_excel_by_price_diff(processed_path)
            if filtered_path == processed_path:
                 # Corrected indentation
                 logger.warning("Price filtering step failed or produced no changes.")
                 # Continue with the current path if filtering failed
                 if not os.path.exists(processed_path):
                     logger.error("File disappeared after filtering attempt.")
                     return None
                 # No change needed: processed_path remains the same
            else:
                 # Corrected indentation
                 logger.info("Successfully filtered by price difference: %s", filtered_path)
                 processed_path = filtered_path # Update path after filtering

        logger.info("Excel file processing completed. Final file: %s", processed_path)
        return processed_path

    # Broad exception for unexpected errors during the overall process
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error(
            # Wrapped long line
            "An unexpected error occurred during Excel processing for '%s': %s",
            input_path,
            str(e),
            exc_info=True, # Log traceback for unexpected errors
        )
        return None
