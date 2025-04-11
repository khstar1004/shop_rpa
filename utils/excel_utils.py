"""
Excel 파일 처리를 위한 유틸리티 함수들

PythonScript 폴더의 Excel 관련 기능을 활용하여 일반적인 Excel 작업을 쉽게 수행할 수 있는
유틸리티 함수들을 제공합니다.
"""

import logging
import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, PatternFill, Side

# 기본 로거 설정
logger = logging.getLogger(__name__)


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
    except Exception as e:
        logger.error("Unexpected error checking Excel file columns: %s", str(e))
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
        except Exception as e: # Catch other potential read errors
            logger.error("Unexpected error reading XLS file %s with read_html: %s", file_path, str(e))
            return ""

        # 첫 번째 행을 헤더로 사용
        if df.empty:
            logger.warning("XLS file %s seems empty after reading.", file_path)
            return ""
        df.columns = df.iloc[0].astype(str).str.strip() # Ensure header is string
        df = df.drop(0).reset_index(drop=True)

        # 상품명 전처리 (// 이후 제거 및 특수 문자 정리)
        pattern = r"(\d{4}_[A-Z]\.)|(\d+\+\d+)|[^a-zA-Z0-9가-힣\s]|\s+"

        def preprocess_product_name(product_name):
            if not isinstance(product_name, str):
                return str(product_name) # Convert non-strings

            # // 이후 제거
            if "//" in product_name:
                product_name = product_name.split("//")[0]

            # 특수 문자 및 패턴 제거
            product_name = re.sub(pattern, " ", product_name)
            product_name = (
                product_name.replace("정품", "").replace("NEW", "").replace("특가", "")
            )
            product_name = product_name.replace("주문제작타올", "").replace(
                "주문제작수건", ""
            )
            product_name = product_name.replace("결혼답례품 수건", "").replace(
                "답례품수건", ""
            )
            product_name = product_name.replace("주문제작 수건", "").replace(
                "돌답례품수건", ""
            )
            product_name = (
                product_name.replace("명절선물세트", "")
                .replace("각종행사수건", "")
                .strip()
            )
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
        logger.error("Directory or file not found during XLS conversion: %s / %s", input_directory, file_name)
        return ""
    except (OSError, IOError) as e:
        logger.error("File system error during XLS conversion: %s", e)
        # Clean up potentially created output file
        if output_file_path and os.path.exists(output_file_path):
             try:
                 os.remove(output_file_path)
             except OSError as rm_err:
                 logger.error("Failed to remove partial XLSX file: %s", rm_err)
        return ""
    except (AttributeError, TypeError, KeyError, IndexError) as e:
         logger.error("Data processing error during XLS conversion (check DataFrame structure): %s", e)
         return ""
    except Exception as e:
        logger.error("Unexpected error converting XLS to XLSX: %s", str(e))
        # Clean up potentially created output file
        if output_file_path and os.path.exists(output_file_path):
             try:
                 os.remove(output_file_path)
             except OSError as rm_err:
                 logger.error("Failed to remove partial XLSX file: %s", rm_err)
        return ""


def add_hyperlinks_to_excel(file_path: str) -> str:
    """
    엑셀 파일의 URL 필드를 하이퍼링크로 변환하고 기본 서식을 적용합니다.
    새로운 파일에 결과를 저장합니다.

    Args:
        file_path: 입력 엑셀 파일 경로

    Returns:
        str: 처리된 새 파일 경로 또는 오류 시 원본 파일 경로
    """
    output_file_path = file_path # Default return on error
    try:
        logger.info("Adding hyperlinks and formatting to Excel file: %s", file_path)

        # 출력 파일명 생성 (기존 파일 덮어쓰지 않음)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_formatted_{timestamp}.xlsx"
        output_file_path = os.path.join(output_directory, output_filename)

        # DataFrame 로드
        df = pd.read_excel(file_path)

        # 워크북 로드 (openpyxl 사용)
        try:
             wb = load_workbook(file_path)
             ws = wb.active
        except FileNotFoundError:
             logger.error("Input file not found for openpyxl: %s", file_path)
             return file_path
        except Exception as e: # Catch potential openpyxl load errors
             logger.error("Error loading workbook with openpyxl: %s", e)
             return file_path

        # 테두리 스타일 정의
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        # 모든 열의 너비 설정 및 자동 줄 바꿈, 테두리 적용
        for col_idx in range(1, ws.max_column + 1):
            col_letter = ws.cell(row=1, column=col_idx).column_letter
            ws.column_dimensions[col_letter].width = 16 # 고정 너비

            # Apply formatting to existing cells
            max_row_to_format = min(ws.max_row, df.shape[0] + 1) # Limit to data rows + header
            for row_idx in range(1, max_row_to_format + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.alignment = Alignment(wrap_text=True, vertical='center') # Added vertical alignment
                cell.border = thin_border

        # 링크 컬럼 처리
        link_columns = [
            "본사상품링크",
            "고려기프트 상품링크",
            "네이버 쇼핑 링크",
            "공급사 상품링크",
        ]
        for link_column in link_columns:
            if link_column in df.columns:
                col_idx = df.columns.get_loc(link_column) + 1
                for row_idx, link in enumerate(
                    df[link_column], start=2
                ):  # 헤더 제외하고 시작
                    # Check link validity before applying hyperlink
                    if (
                        pd.notna(link)
                        and isinstance(link, str)
                        and (link.startswith("http:") or link.startswith("https:"))
                    ):
                        try:
                            cell = ws.cell(row=row_idx, column=col_idx)
                            # Set value and hyperlink only if cell exists
                            if cell:
                                 cell.value = link
                                 cell.hyperlink = link
                        except (AttributeError, IndexError) as cell_err:
                             logger.warning("Could not access cell at row %d, col %d for hyperlink: %s", row_idx, col_idx, cell_err)
                             continue # Skip this cell

        # 헤더 색상 적용
        gray_fill = PatternFill(
            start_color="CCCCCC", end_color="CCCCCC", fill_type="solid"
        )
        for cell in ws["1:1"]: # Format first row (header)
            cell.fill = gray_fill
            cell.font = cell.font.copy(bold=True) # Make header bold

        # 가격차이가 음수인 행에 노란색 배경 적용
        yellow_fill = PatternFill(
            start_color="FFFF00", end_color="FFFF00", fill_type="solid"
        )
        price_diff_cols = ["가격차이(2)", "가격차이(3)"]
        col_indices = {ws.cell(row=1, column=j+1).value: j for j in range(ws.max_column)}

        for row_idx in range(2, df.shape[0] + 2):
             highlight_row = False
             for col_name in price_diff_cols:
                 col_idx = col_indices.get(col_name)
                 if col_idx is not None:
                     try:
                         value = df.loc[row_idx - 2, col_name] # Access DataFrame for value check
                         if pd.notna(value):
                             value = float(value) # Convert to float for comparison
                             if value < 0:
                                 highlight_row = True
                                 break
                     except (ValueError, TypeError, KeyError, IndexError) as data_err:
                         # Ignore errors if data cannot be converted/accessed
                         # logger.debug("Could not check price diff value at row %d, col %s: %s", row_idx, col_name, data_err)
                         continue

             if highlight_row:
                 for col in range(1, ws.max_column + 1):
                      try:
                          ws.cell(row=row_idx, column=col).fill = yellow_fill
                      except (AttributeError, IndexError) as cell_err:
                           logger.warning("Could not apply fill to cell at row %d, col %d: %s", row_idx, col, cell_err)
                           continue

        # 결과 저장 (새 파일에)
        wb.save(output_file_path)
        logger.info("Excel file with formatting saved to: %s", output_file_path)

        return output_file_path

    except FileNotFoundError:
        logger.error("Input Excel file not found: %s", file_path)
        return file_path
    except (pd.errors.ParserError, ValueError) as e:
        logger.error("Error parsing input Excel file with Pandas: %s", e)
        return file_path
    except ImportError:
         logger.error("Required library (pandas or openpyxl) not found.")
         return file_path
    except (OSError, IOError) as e:
        logger.error("File system error during Excel processing/saving: %s", e)
        # Attempt to clean up partially created output file
        if output_file_path != file_path and os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
            except OSError as remove_err:
                logger.warning("Could not remove partially created file %s: %s", output_file_path, remove_err)
        return file_path
    except Exception as e:
        logger.error("Unexpected error adding hyperlinks/formatting: %s", str(e))
        # Attempt to clean up partially created output file
        if output_file_path != file_path and os.path.exists(output_file_path):
             try:
                 os.remove(output_file_path)
             except OSError:
                 pass # Ignore cleanup error
        return file_path


def filter_excel_by_price_diff(file_path: str) -> str:
    """
    가격차이가 있는 항목들만 필터링하여 업로드용 엑셀 파일을 생성합니다.
    새로운 파일에 결과를 저장합니다.

    Args:
        file_path: 입력 엑셀 파일 경로

    Returns:
        str: 필터링된 출력 파일 경로 또는 오류 시 원본 파일 경로
    """
    output_path = file_path # Default return on error
    try:
        logger.info("Filtering Excel file by price differences: %s", file_path)

        # 출력 파일명 생성 (새 파일에 저장)
        input_filename = os.path.splitext(os.path.basename(file_path))[0]
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_filename}_upload_{current_datetime}.xlsx"
        output_directory = os.path.dirname(file_path)
        output_path = os.path.join(output_directory, output_filename)

        # 데이터프레임 로드
        df = pd.read_excel(file_path)

        # 문자열을 숫자로 안전하게 변환 (to_float 함수 정의는 동일하게 유지)
        def to_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        if "가격차이(2)" in df.columns:
            df["가격차이(2)"] = df["가격차이(2)"].apply(to_float)
        if "가격차이(3)" in df.columns:
            df["가격차이(3)"] = df["가격차이(3)"].apply(to_float)

        # 가격차이가 음수인 항목만 필터링
        try:
            price_diff_condition_2 = df["가격차이(2)"].notna() & (df["가격차이(2)"] < 0) if "가격차이(2)" in df.columns else False
            price_diff_condition_3 = df["가격차이(3)"].notna() & (df["가격차이(3)"] < 0) if "가격차이(3)" in df.columns else False
            combined_condition = price_diff_condition_2 | price_diff_condition_3
        except TypeError as e:
            logger.warning("Could not evaluate price difference condition due to data type issues: %s", e)
            return file_path # Return original if condition cannot be evaluated

        if not isinstance(combined_condition, pd.Series) or not combined_condition.any():
             logger.warning("No items found with negative price differences.")
             # Return original path instead of creating an empty file
             return file_path

        filtered_df = df[combined_condition].copy() # Create a copy to avoid SettingWithCopyWarning

        # --- 컬럼 선택 및 이름 변경 (기존 코드 유지) ---
        required_columns = [
            "구분", "담당자", "업체명", "업체코드", "Code", "중분류카테고리", "상품명",
            "기본수량(1)", "판매단가(V포함)", "본사상품링크",
            "기본수량(2)", "판매단가(V포함)(2)", "가격차이(2)", "가격차이(2)(%)", "고려기프트 상품링크",
            "기본수량(3)", "판매단가(V포함)(3)", "가격차이(3)", "가격차이(3)(%)", "공급사명",
            "공급사 상품링크", "본사 이미지", "고려기프트 이미지", "네이버 이미지",
        ]
        existing_columns = [col for col in required_columns if col in filtered_df.columns]
        filtered_df = filtered_df[existing_columns]
        column_mapping = {
            "구분": "구분(승인관리:A/가격관리:P)", "담당자": "담당자", "업체명": "공급사명",
            "업체코드": "공급처코드", "Code": "상품코드", "중분류카테고리": "카테고리(중분류)",
            "상품명": "상품명", "기본수량(1)": "본사 기본수량", "판매단가(V포함)": "판매단가1(VAT포함)",
            "본사상품링크": "본사링크", "기본수량(2)": "고려 기본수량",
            "판매단가(V포함)(2)": "판매단가2(VAT포함)", "가격차이(2)": "고려 가격차이",
            "가격차이(2)(%)": "고려 가격차이(%)", "고려기프트 상품링크": "고려 링크",
            "기본수량(3)": "네이버 기본수량", "판매단가(V포함)(3)": "판매단가3 (VAT포함)",
            "가격차이(3)": "네이버 가격차이", "가격차이(3)(%)": "네이버가격차이(%)",
            "공급사명": "네이버 공급사명", "공급사 상품링크": "네이버 링크",
            "본사 이미지": "해오름(이미지링크)", "고려기프트 이미지": "고려기프트(이미지링크)",
            "네이버 이미지": "네이버쇼핑(이미지링크)",
        }
        rename_mapping = {k: v for k, v in column_mapping.items() if k in filtered_df.columns}
        filtered_df.rename(columns=rename_mapping, inplace=True)
        # --- 컬럼 선택 및 이름 변경 끝 ---

        # 엑셀로 저장
        filtered_df.to_excel(output_path, index=False)

        # 추가 포맷팅 적용 (openpyxl 사용)
        wb = load_workbook(output_path)
        ws = wb.active

        # --- 스타일 설정 및 적용 (기존 코드 유지) ---
        thin_border = Border(left=Side(style="thin"), right=Side(style="thin"), top=Side(style="thin"), bottom=Side(style="thin"))
        # 테두리 및 정렬 적용
        for row in ws.iter_rows(min_row=2, max_col=ws.max_column, max_row=ws.max_row):
            for cell in row:
                cell.border = thin_border
        for col in ws.iter_cols(min_col=1, max_col=ws.max_column):
            for cell in col:
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        # 컬럼 너비 및 행 높이 설정
        for col in ws.iter_cols(min_col=1, max_col=ws.max_column, max_row=1):
            for cell in col:
                ws.column_dimensions[cell.column_letter].width = 15
        for row in ws.iter_rows(min_row=1, max_col=ws.max_column, max_row=ws.max_row):
            for cell in row:
                ws.row_dimensions[cell.row].height = 16.5
        # --- 스타일 설정 및 적용 끝 ---

        # 품절 항목 및 특정 조건 처리 (내부 함수 호출)
        _apply_filter_rules(ws)

        # 변경사항 저장
        wb.save(output_path)
        logger.info("Filtered Excel file saved to: %s", output_path)

        return output_path

    except FileNotFoundError:
        logger.error("Input file not found for filtering: %s", file_path)
        return file_path
    except (pd.errors.ParserError, ValueError) as e:
        logger.error("Error parsing input Excel file with Pandas: %s", e)
        return file_path
    except ImportError:
         logger.error("Required library (pandas or openpyxl) not found.")
         return file_path
    except KeyError as e:
        logger.error("Missing expected column during filtering or renaming: %s", e)
        return file_path
    except (OSError, IOError) as e:
        logger.error("File system error during filtering/saving: %s", e)
        if output_path != file_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as rm_err:
                logger.warning("Could not remove partially created file %s: %s", output_path, rm_err)
        return file_path
    except Exception as e:
        logger.error("Unexpected error filtering Excel by price differences: %s", str(e))
        if output_path != file_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as rm_err:
                logger.warning("Could not remove partially created file %s: %s", output_path, rm_err)
        return file_path


def _apply_filter_rules(worksheet) -> None:
    """
    주어진 워크시트에 필터링 규칙(품절 행 제거, 유효하지 않은 링크 제거)을 적용합니다.
    openpyxl 워크시트 객체를 직접 수정합니다.

    Args:
        worksheet: 처리할 openpyxl 워크시트 객체
    """
    try:
        from urllib.parse import urlparse
    except ImportError:
        logger.error("Could not import urlparse. Skipping link validation.")
        urlparse = None # Set to None to disable link validation if import fails

    try:
        # 컬럼 인덱스를 미리 찾아둡니다.
        header = [cell.value for cell in worksheet[1]] # Get header row values
        col_indices = {name: i for i, name in enumerate(header) if name}

        goleo_quantity_col_idx = col_indices.get("고려 기본수량")
        naver_quantity_col_idx = col_indices.get("네이버 기본수량")
        goleo_link_col_idx = col_indices.get("고려 링크")
        naver_link_col_idx = col_indices.get("네이버 링크")

        rows_to_delete = []
        # 워크시트의 최대 행부터 역순으로 순회합니다.
        for row_idx in range(worksheet.max_row, 1, -1):
            row_cells = list(worksheet[row_idx]) # Get cells for the current row

            # 1. 품절 항목 삭제
            delete_this_row = False
            if goleo_quantity_col_idx is not None:
                 cell_value = row_cells[goleo_quantity_col_idx].value
                 if cell_value == "품절":
                     delete_this_row = True

            if not delete_this_row and naver_quantity_col_idx is not None:
                 cell_value = row_cells[naver_quantity_col_idx].value
                 if cell_value == "품절":
                     delete_this_row = True

            if delete_this_row:
                rows_to_delete.append(row_idx)
                continue # 다음 행으로 이동

            # 2. 링크 검증 (urlparse 임포트 성공 시)
            if urlparse:
                if goleo_link_col_idx is not None:
                    link_cell = row_cells[goleo_link_col_idx]
                    link_value = link_cell.value
                    if (
                        link_value
                        and isinstance(link_value, str)
                        and not bool(urlparse(link_value).scheme)
                    ):
                        link_cell.value = None # 유효하지 않으면 None으로 설정

                if naver_link_col_idx is not None:
                    link_cell = row_cells[naver_link_col_idx]
                    link_value = link_cell.value
                    if (
                        link_value
                        and isinstance(link_value, str)
                        and not bool(urlparse(link_value).scheme)
                    ):
                        link_cell.value = None

        # 찾은 행들을 한 번에 삭제 (역순으로 삭제해야 인덱스 문제 없음)
        if rows_to_delete:
            for row_idx in sorted(rows_to_delete, reverse=True):
                 worksheet.delete_rows(row_idx)
            logger.info("Removed %d rows based on filter rules.", len(rows_to_delete))

    except (AttributeError, IndexError, TypeError, ValueError) as e:
        logger.error("Error applying filter rules to worksheet: %s", e)
    except Exception as e:
        logger.error("Unexpected error applying filter rules: %s", e)


def insert_image_to_excel(image_path: str, target_cell: str) -> bool:
    """
    이미지를 엑셀 셀에 삽입합니다. (Windows 전용, win32com 필요)

    Args:
        image_path: 이미지 파일 경로
        target_cell: 이미지를 삽입할 셀 주소 (예: "A1")

    Returns:
        bool: 성공 여부
    """
    resized_image_path = None # For cleanup on error
    try:
        logger.info("Attempting to insert image %s to cell %s", image_path, target_cell)

        # Import necessary libraries (only when function is called)
        try:
            import win32com.client as win32
            from PIL import Image as PILImage
        except ImportError as import_err:
            logger.error(
                "Required libraries missing (win32com, pillow): %s. Image insertion skipped.", import_err
            )
            return False

        # Validate image path
        try:
            abs_image_path = os.path.abspath(image_path)
            if not os.path.exists(abs_image_path):
                logger.error("Image file not found: %s", abs_image_path)
                return False
        except OSError as e:
            logger.error("Error accessing image path %s: %s", image_path, e)
            return False

        # Resize image using PIL
        try:
            with PILImage.open(abs_image_path) as img:
                img = img.convert("RGB")  # Remove transparency
                img_resized = img.resize((100, 100)) # Target size
                base, ext = os.path.splitext(abs_image_path)
                resized_image_path = f"{base}_resized{ext}"
                img_resized.save(resized_image_path)
                logger.debug("Resized image saved to: %s", resized_image_path)
        except FileNotFoundError:
             logger.error("Image file not found during PIL open: %s", abs_image_path)
             return False
        except (PILImage.UnidentifiedImageError, ValueError, IOError) as pil_err:
            logger.error("Error processing image with PIL: %s", pil_err)
            return False
        except Exception as e: # Catch-all for other PIL errors
            logger.error("Unexpected error resizing image: %s", str(e))
            return False

        # Validate target cell address format
        if not re.match("^[A-Z]+[0-9]+$", target_cell):
            logger.error("Invalid cell address format: %s", target_cell)
            # Clean up resized image before returning
            if resized_image_path and os.path.exists(resized_image_path):
                 try:
                     os.remove(resized_image_path)
                 except OSError as rm_err:
                     logger.warning("Could not remove resized image file %s after invalid cell format: %s", resized_image_path, rm_err)
            return False

        # Interact with Excel using win32com
        excel = None
        workbook = None
        try:
            # Get or open Excel application
            try:
                 excel = win32.GetActiveObject("Excel.Application")
                 logger.debug("Connected to active Excel instance.")
            except Exception:
                 logger.warning("No active Excel instance found. Attempting to start Excel.")
                 try:
                      excel = win32.Dispatch("Excel.Application")
                      excel.Visible = True # Make it visible for debugging if needed
                      logger.info("Started new Excel instance.")
                 except Exception as dispatch_err:
                      logger.error("Failed to start Excel application: %s", dispatch_err)
                      # Clean up resized image
                      if resized_image_path and os.path.exists(resized_image_path):
                           try:
                               os.remove(resized_image_path)
                           except OSError as rm_err:
                               logger.warning("Could not remove resized image file %s after Excel start failure: %s", resized_image_path, rm_err)
                      return False

            # Get the active workbook and worksheet
            try:
                 workbook = excel.ActiveWorkbook
                 if not workbook:
                      logger.error("No active workbook found in Excel.")
                      if resized_image_path and os.path.exists(resized_image_path):
                           try:
                               os.remove(resized_image_path)
                           except OSError as rm_err:
                               logger.warning("Could not remove resized image file %s after no active workbook: %s", resized_image_path, rm_err)
                      return False
                 worksheet = workbook.Worksheets(1) # Assuming first worksheet
            except Exception as wb_err:
                 logger.error("Error accessing active workbook/worksheet: %s", wb_err)
                 if resized_image_path and os.path.exists(resized_image_path):
                     try:
                         os.remove(resized_image_path)
                     except OSError as rm_err:
                         logger.warning("Could not remove resized image file %s after workbook access error: %s", resized_image_path, rm_err)
                 return False

            # Extract column and row from target cell
            match = re.match(r"([A-Z]+)(\d+)", target_cell)
            if not match:
                 # This case should be caught by the earlier regex, but double-check
                 logger.error("Could not parse cell address: %s", target_cell)
                 if resized_image_path and os.path.exists(resized_image_path):
                     try:
                         os.remove(resized_image_path)
                     except OSError as rm_err:
                          logger.warning("Could not remove resized image file %s after cell parse error: %s", resized_image_path, rm_err)
                 return False
            col_letter = match.group(1)
            row_number = int(match.group(2))

            # Adjust cell size for image
            worksheet.Cells(row_number, col_letter).ColumnWidth = 100 / 6.25 # Approx conversion factor
            worksheet.Cells(row_number, col_letter).RowHeight = 100

            # Insert the picture
            start_cell = worksheet.Range(target_cell)
            left = start_cell.Left
            top = start_cell.Top

            worksheet.Shapes.AddPicture(
                resized_image_path, # Use the resized image path
                LinkToFile=False,
                SaveWithDocument=True,
                Left=left,
                Top=top,
                Width=100,
                Height=100,
            )

            # Save the workbook
            workbook.Save()
            logger.info("Image inserted successfully into cell %s", target_cell)

            # Clean up the resized image file after successful insertion
            if resized_image_path and os.path.exists(resized_image_path):
                try:
                    os.remove(resized_image_path)
                    logger.debug("Cleaned up resized image: %s", resized_image_path)
                except OSError as rm_err:
                    logger.warning("Could not remove resized image file %s: %s", resized_image_path, rm_err)

            return True

        except AttributeError as ae:
            # Commonly occurs if COM object methods/properties are wrong
            logger.error("COM object error (check Excel/win32 interaction): %s", ae)
            if resized_image_path and os.path.exists(resized_image_path):
                try:
                    os.remove(resized_image_path)
                except OSError as rm_err:
                     logger.warning("Could not remove resized image file %s after COM error: %s", resized_image_path, rm_err)
            return False
        except ValueError as ve:
            # e.g., invalid worksheet index
            logger.error("Value error during Excel interaction: %s", ve)
            if resized_image_path and os.path.exists(resized_image_path):
                try:
                    os.remove(resized_image_path)
                except OSError as rm_err:
                     logger.warning("Could not remove resized image file %s after value error: %s", resized_image_path, rm_err)
            return False
        except Exception as e:
            # Catch other COM or Excel related errors
            logger.error("Error interacting with Excel via win32com: %s", str(e))
            if resized_image_path and os.path.exists(resized_image_path):
                try:
                    os.remove(resized_image_path)
                except OSError as rm_err:
                     logger.warning("Could not remove resized image file %s after win32com error: %s", resized_image_path, rm_err)
            return False

    except Exception as e:
        # Top-level catch for unexpected errors in the function logic itself
        logger.error("Unexpected error in insert_image_to_excel function: %s", str(e))
        if resized_image_path and os.path.exists(resized_image_path):
            try:
                os.remove(resized_image_path)
            except OSError as rm_err:
                 logger.warning("Could not remove resized image file %s after top-level error: %s", resized_image_path, rm_err)
        return False


def process_excel_file(input_path: str) -> Optional[str]:
    """
    엑셀 파일을 처리하는 전체 프로세스를 실행합니다:
    XLS 변환 -> 컬럼 확인/추가 -> 하이퍼링크/서식 적용 -> 가격 차이 필터링.

    Args:
        input_path: 입력 엑셀 파일 경로

    Returns:
        Optional[str]: 최종 처리된 파일 경로 또는 오류 시 None
    """
    current_file_path = input_path
    try:
        # 파일 경로 확인
        if not os.path.exists(current_file_path):
            logger.error("Input file not found: %s", current_file_path)
            return None

        # 1. XLS -> XLSX 변환 (확장자가 .xls인 경우)
        input_ext = os.path.splitext(current_file_path)[1].lower()
        input_dir = os.path.dirname(current_file_path)
        original_file_was_xls = False

        if input_ext == ".xls":
            original_file_was_xls = True
            logger.info("XLS file detected: %s", current_file_path)
            xlsx_file = convert_xls_to_xlsx(input_dir)
            if xlsx_file:
                logger.info("XLS file converted to XLSX: %s", xlsx_file)
                current_file_path = xlsx_file # Use the converted file for subsequent steps
            else:
                # If conversion fails, stop processing
                logger.error("XLS to XLSX conversion failed. Stopping processing.")
                return None

        # 2. 필요한 컬럼 확인 및 추가
        if not check_excel_columns(current_file_path):
             logger.error("Failed to check or add required columns. Stopping processing.")
             # Clean up converted file if conversion happened
             if original_file_was_xls and os.path.exists(current_file_path):
                  try:
                      os.remove(current_file_path)
                  except OSError as rm_err:
                       logger.warning("Could not remove converted file %s: %s", current_file_path, rm_err)
             return None

        # 3. 하이퍼링크 추가 및 기본 서식 적용 (새 파일 생성)
        formatted_file = add_hyperlinks_to_excel(current_file_path)
        if formatted_file == current_file_path:
             logger.error("Failed to add hyperlinks/formatting. Stopping processing.")
             # Clean up converted file if conversion happened
             if original_file_was_xls and os.path.exists(current_file_path):
                  try:
                      os.remove(current_file_path)
                  except OSError as rm_err:
                       logger.warning("Could not remove converted file %s after format fail: %s", current_file_path, rm_err)
             return None
        # If formatting created a new file, update current_file_path
        # and potentially remove the intermediate file (like converted xlsx)
        if formatted_file != current_file_path:
             if original_file_was_xls and os.path.exists(current_file_path):
                  logger.debug("Removing intermediate file: %s", current_file_path)
                  try:
                      os.remove(current_file_path)
                  except OSError as rm_err:
                       logger.warning("Could not remove intermediate converted file %s: %s", current_file_path, rm_err)
             current_file_path = formatted_file

        # 4. 가격 차이 필터링 (새 파일 생성)
        filtered_file = filter_excel_by_price_diff(current_file_path)
        if filtered_file == current_file_path:
             # This means either no filtering was needed or an error occurred during filtering
             # If no filtering needed, the 'formatted_file' is the final result.
             # If error, the function `filter_excel_by_price_diff` returns the input path.
             logger.info("Price difference filtering resulted in no changes or an error occurred. Returning the formatted file.")
             # No need to remove current_file_path here as it's the intended result or needed for debugging.
             return current_file_path

        # If filtering created a new file, remove the previous intermediate file
        if filtered_file != current_file_path and os.path.exists(current_file_path):
             logger.debug("Removing intermediate file: %s", current_file_path)
             try:
                 os.remove(current_file_path)
             except OSError as rm_err:
                 logger.warning("Could not remove intermediate formatted file %s: %s", current_file_path, rm_err)

        logger.info("Excel file processing complete. Final file: %s", filtered_file)
        return filtered_file

    except FileNotFoundError as e:
        # Catch cases where the file path becomes invalid mid-process
        logger.error("File not found during processing pipeline: %s", e)
        return None
    except (pd.errors.ParserError, ValueError, ImportError) as e:
        # Catch general parsing/dependency errors from called functions if they propagate
        logger.error("Data parsing or dependency error during processing: %s", e)
        return None
    except (OSError, IOError) as e:
        logger.error("File system error during overall processing: %s", e)
        return None
    except Exception as e:
        # General fallback for unexpected errors in the pipeline orchestration
        logger.error("Unexpected error during Excel file processing pipeline: %s", str(e))
        return None
