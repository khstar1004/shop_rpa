"""
Excel 파일 처리를 위한 유틸리티 함수들

PythonScript 폴더의 Excel 관련 기능을 활용하여 일반적인 Excel 작업을 쉽게 수행할 수 있는 
유틸리티 함수들을 제공합니다.
"""

import os
import logging
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Alignment, Border, Side
from urllib.parse import urlparse
from datetime import datetime
import re
from typing import List, Optional

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
        logger.info(f"Checking Excel file: {file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            logger.error(f"Excel file not found: {file_path}")
            return False
        
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)
        
        # 추가할 컬럼 정의
        columns_to_add = ['본사 이미지', '고려기프트 이미지', '네이버 이미지']
        need_to_modify = False
        
        # 컬럼 확인 및 추가
        for column in columns_to_add:
            if column not in df.columns:
                df[column] = ''
                need_to_modify = True
                logger.info(f"Added missing column: {column}")
        
        if not need_to_modify:
            logger.info("All required columns exist. No modifications needed.")
            return True
        
        # 컬럼명의 앞뒤 공백 제거
        df.columns = [col.strip() for col in df.columns]
        
        # 엑셀 파일 저장
        df.to_excel(file_path, index=False)
        logger.info(f"Updated Excel file with required columns: {file_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error checking Excel file: {str(e)}")
        return False

def convert_xls_to_xlsx(input_directory: str) -> str:
    """
    XLS 파일을 XLSX 형식으로 변환합니다.
    
    Args:
        input_directory: 입력 디렉토리 경로
        
    Returns:
        str: 변환된 XLSX 파일 경로 또는 빈 문자열
    """
    try:
        logger.info(f"Looking for XLS files in: {input_directory}")
        
        # XLS 파일 찾기
        xls_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.xls')]
        
        if not xls_files:
            logger.warning("No XLS files found in the directory.")
            return ""
        
        # 첫 번째 XLS 파일 처리
        file_name = xls_files[0]
        file_path = os.path.join(input_directory, file_name)
        
        logger.info(f"Converting XLS file: {file_path}")
        
        # XLS 파일 로드
        try:
            tables = pd.read_html(file_path, encoding='cp949')
            df = tables[0]
        except Exception as e:
            logger.error(f"Error reading XLS file: {str(e)}")
            return ""
        
        # 첫 번째 행을 헤더로 사용
        df.columns = df.iloc[0].str.strip()
        df = df.drop(0)
        
        # 상품명 전처리 (// 이후 제거 및 특수 문자 정리)
        pattern = r'(\d{4}_[A-Z]\.)|(\d+\+\d+)|[^a-zA-Z0-9가-힣\s]|\s+'
        
        def preprocess_product_name(product_name):
            if not isinstance(product_name, str):
                return product_name
            
            # // 이후 제거
            if "//" in product_name:
                product_name = product_name.split("//")[0]
            
            # 특수 문자 및 패턴 제거
            product_name = re.sub(pattern, ' ', product_name)
            product_name = product_name.replace('정품', '').replace('NEW', '').replace('특가', '')
            product_name = product_name.replace('주문제작타올', '').replace('주문제작수건', '')
            product_name = product_name.replace('결혼답례품 수건', '').replace('답례품수건', '')
            product_name = product_name.replace('주문제작 수건', '').replace('돌답례품수건', '')
            product_name = product_name.replace('명절선물세트', '').replace('각종행사수건', '').strip()
            product_name = re.sub(' +', ' ', product_name)
            return product_name
        
        if '상품명' in df.columns:
            df['상품명'] = df['상품명'].apply(preprocess_product_name)
        
        # 필요한 컬럼 추가
        for column in ['본사 이미지', '고려기프트 이미지', '네이버 이미지']:
            if column not in df.columns:
                df[column] = ''
        
        # 출력 파일명 설정
        output_file_name = file_name.replace('.xls', '.xlsx')
        output_file_path = os.path.join(input_directory, output_file_name)
        
        # XLSX로 저장
        df.to_excel(output_file_path, index=False)
        logger.info(f"Converted file saved to: {output_file_path}")
        
        return output_file_path
        
    except Exception as e:
        logger.error(f"Error converting XLS to XLSX: {str(e)}")
        return ""

def add_hyperlinks_to_excel(file_path: str) -> str:
    """
    엑셀 파일의 URL 필드를 하이퍼링크로 변환합니다.
    
    Args:
        file_path: 입력 엑셀 파일 경로
        
    Returns:
        str: 처리된 파일 경로
    """
    try:
        logger.info(f"Adding hyperlinks to Excel file: {file_path}")
        
        # 출력 파일명 생성
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        output_directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_result_{timestamp}.xlsx"
        output_file_path = os.path.join(output_directory, output_filename)
        
        # DataFrame 로드
        df = pd.read_excel(file_path)
        
        # 워크북 로드
        wb = load_workbook(file_path)
        ws = wb.active
        
        # 테두리 스타일 정의
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # 모든 열의 너비 설정 및 자동 줄 바꿈, 테두리 적용
        for col in range(1, ws.max_column + 1):
            col_letter = ws.cell(row=1, column=col).column_letter
            ws.column_dimensions[col_letter].width = 16
            
            for row in range(1, df.shape[0] + 2):  # 첫 번째 행은 헤더
                cell = ws.cell(row=row, column=col)
                cell.alignment = Alignment(wrap_text=True)
                cell.border = thin_border
        
        # 링크 컬럼 처리
        link_columns = ["본사상품링크", "고려기프트 상품링크", "네이버 쇼핑 링크", "공급사 상품링크"]
        for link_column in link_columns:
            if link_column in df.columns:
                col_idx = df.columns.get_loc(link_column) + 1
                for row_idx, link in enumerate(df[link_column], start=2):  # 첫 번째 행은 헤더
                    if pd.notna(link) and isinstance(link, str) and (link.startswith("http") or link.startswith("https")):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        cell.value = link
                        cell.hyperlink = link
        
        # 헤더 색상 적용
        gray_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
        for cell in ws["1:1"]:
            cell.fill = gray_fill
        
        # 가격차이가 음수인 행에 노란색 배경 적용
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        for row_idx in range(2, df.shape[0] + 2):
            for col_name in ["가격차이(2)", "가격차이(3)"]:
                if col_name in df.columns:
                    try:
                        value = df.loc[row_idx - 2, col_name]
                        if pd.notna(value):
                            value = float(value)
                            if value < 0:
                                for col in ws.iter_cols(min_row=row_idx, max_row=row_idx, min_col=1, max_col=ws.max_column):
                                    for cell in col:
                                        cell.fill = yellow_fill
                                break
                    except (ValueError, TypeError):
                        continue
        
        # 결과 저장
        wb.save(output_file_path)
        logger.info(f"Excel file with hyperlinks saved to: {output_file_path}")
        
        return output_file_path
        
    except Exception as e:
        logger.error(f"Error adding hyperlinks to Excel: {str(e)}")
        return file_path

def filter_excel_by_price_diff(file_path: str) -> str:
    """
    가격차이가 있는 항목들만 필터링하여 업로드용 엑셀 파일을 생성합니다.
    
    Args:
        file_path: 입력 엑셀 파일 경로
        
    Returns:
        str: 필터링된 출력 파일 경로
    """
    try:
        logger.info(f"Filtering Excel file by price differences: {file_path}")
        
        # 출력 파일명 생성
        input_filename = os.path.splitext(os.path.basename(file_path))[0]
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{input_filename}_upload_{current_datetime}.xlsx"
        output_directory = os.path.dirname(file_path)
        output_path = os.path.join(output_directory, output_filename)
        
        # 데이터프레임 로드
        df = pd.read_excel(file_path)
        
        # 문자열을 숫자로 변환
        def to_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        if '가격차이(2)' in df.columns:
            df['가격차이(2)'] = df['가격차이(2)'].apply(to_float)
        if '가격차이(3)' in df.columns:
            df['가격차이(3)'] = df['가격차이(3)'].apply(to_float)
        
        # 가격차이가 음수인 항목만 필터링
        price_diff_condition = False
        if '가격차이(2)' in df.columns:
            price_diff_condition |= (df['가격차이(2)'].notna() & (df['가격차이(2)'] < 0))
        if '가격차이(3)' in df.columns:
            price_diff_condition |= (df['가격차이(3)'].notna() & (df['가격차이(3)'] < 0))
        
        if isinstance(price_diff_condition, bool) and not price_diff_condition:
            logger.warning("No price difference columns found or no conditions specified.")
            return file_path
        
        filtered_df = df[price_diff_condition]
        
        if filtered_df.empty:
            logger.warning("No items with negative price differences found.")
            return file_path
        
        # 원하는 컬럼만 선택
        required_columns = [
            '구분', '담당자', '업체명', '업체코드', 'Code', '중분류카테고리', '상품명', 
            '기본수량(1)', '판매단가(V포함)', '본사상품링크',
            '기본수량(2)', '판매단가(V포함)(2)', '가격차이(2)', '가격차이(2)(%)', '고려기프트 상품링크',
            '기본수량(3)', '판매단가(V포함)(3)', '가격차이(3)', '가격차이(3)(%)', '공급사명', '공급사 상품링크',
            '본사 이미지', '고려기프트 이미지', '네이버 이미지'
        ]
        
        # 존재하는 컬럼만 선택
        existing_columns = [col for col in required_columns if col in filtered_df.columns]
        filtered_df = filtered_df[existing_columns]
        
        # 컬럼 이름 변경
        column_mapping = {
            '구분': '구분(승인관리:A/가격관리:P)',
            '담당자': '담당자',
            '업체명': '공급사명',
            '업체코드': '공급처코드',
            'Code': '상품코드',
            '중분류카테고리': '카테고리(중분류)',
            '상품명': '상품명',
            '기본수량(1)': '본사 기본수량',
            '판매단가(V포함)': '판매단가1(VAT포함)',
            '본사상품링크': '본사링크',
            '기본수량(2)': '고려 기본수량',
            '판매단가(V포함)(2)': '판매단가2(VAT포함)',
            '가격차이(2)': '고려 가격차이',
            '가격차이(2)(%)': '고려 가격차이(%)',
            '고려기프트 상품링크': '고려 링크',
            '기본수량(3)': '네이버 기본수량',
            '판매단가(V포함)(3)': '판매단가3 (VAT포함)',
            '가격차이(3)': '네이버 가격차이',
            '가격차이(3)(%)': '네이버가격차이(%)',
            '공급사명': '네이버 공급사명',
            '공급사 상품링크': '네이버 링크',
            '본사 이미지': '해오름(이미지링크)',
            '고려기프트 이미지': '고려기프트(이미지링크)',
            '네이버 이미지': '네이버쇼핑(이미지링크)'
        }
        
        # 존재하는 컬럼만 매핑
        rename_mapping = {k: v for k, v in column_mapping.items() if k in filtered_df.columns}
        filtered_df.rename(columns=rename_mapping, inplace=True)
        
        # 엑셀로 저장
        filtered_df.to_excel(output_path, index=False)
        
        # 추가 포맷팅 적용
        wb = load_workbook(output_path)
        ws = wb.active
        
        # 스타일 설정
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # 테두리 및 정렬 적용
        for row in ws.iter_rows(min_row=2, max_col=ws.max_column, max_row=ws.max_row):
            for cell in row:
                cell.border = thin_border
        
        for col in ws.iter_cols(min_col=1, max_col=ws.max_column):
            for cell in col:
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        
        # 컬럼 너비 및 행 높이 설정
        for col in ws.iter_cols(min_col=1, max_col=ws.max_column, max_row=1):
            for cell in col:
                ws.column_dimensions[cell.column_letter].width = 15
        
        for row in ws.iter_rows(min_row=1, max_col=ws.max_column, max_row=ws.max_row):
            for cell in row:
                ws.row_dimensions[cell.row].height = 16.5
        
        # 품절 항목 및 특정 조건 처리
        _apply_filter_rules(ws)
        
        # 변경사항 저장
        wb.save(output_path)
        logger.info(f"Filtered Excel file saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error filtering Excel by price differences: {str(e)}")
        return file_path

def _apply_filter_rules(worksheet) -> None:
    """필터링 규칙을 적용합니다."""
    from urllib.parse import urlparse
    
    # 품절 행 삭제 및 데이터 정리 로직
    for row in reversed(list(worksheet.iter_rows(min_row=2, max_row=worksheet.max_row))):
        # 고려 기본수량이 품절일 경우 처리
        goleo_quantity_col = None
        naver_quantity_col = None
        
        # 컬럼 인덱스 찾기
        for i, cell in enumerate(worksheet[1]):
            if cell.value == '고려 기본수량':
                goleo_quantity_col = i
            elif cell.value == '네이버 기본수량':
                naver_quantity_col = i
        
        # 품절 항목 삭제
        if goleo_quantity_col is not None and row[goleo_quantity_col].value == "품절":
            worksheet.delete_rows(row[0].row)
            continue
        
        if naver_quantity_col is not None and row[naver_quantity_col].value == "품절":
            worksheet.delete_rows(row[0].row)
            continue
        
        # 링크 검증
        goleo_link_col = None
        naver_link_col = None
        
        for i, cell in enumerate(worksheet[1]):
            if cell.value == '고려 링크':
                goleo_link_col = i
            elif cell.value == '네이버 링크':
                naver_link_col = i
        
        # 유효하지 않은 링크 제거
        if goleo_link_col is not None:
            link_value = row[goleo_link_col].value
            if link_value and isinstance(link_value, str) and not bool(urlparse(link_value).scheme):
                row[goleo_link_col].value = None
        
        if naver_link_col is not None:
            link_value = row[naver_link_col].value
            if link_value and isinstance(link_value, str) and not bool(urlparse(link_value).scheme):
                row[naver_link_col].value = None

def insert_image_to_excel(image_path: str, target_cell: str) -> bool:
    """
    이미지를 엑셀 셀에 삽입합니다.
    
    Args:
        image_path: 이미지 파일 경로
        target_cell: 이미지를 삽입할 셀 주소 (예: "A1")
    
    Returns:
        bool: 성공 여부
    """
    try:
        logger.info(f"Inserting image {image_path} to cell {target_cell}")
        
        # win32com.client가 필요합니다.
        # 이 기능은 Windows 환경에서만 작동합니다.
        try:
            import win32com.client as win32
            from PIL import Image as PILImage
        except ImportError:
            logger.error("Required libraries missing. Install win32com.client and pillow.")
            return False
        
        # 절대 경로 확인
        abs_image_path = os.path.abspath(image_path)
        
        # 이미지 파일 존재 여부 검사
        if not os.path.exists(abs_image_path):
            logger.error(f"Image file not found: {abs_image_path}")
            return False
        
        # 이미지 크기를 100x100으로 조정
        try:
            with PILImage.open(abs_image_path) as img:
                img = img.convert("RGB")  # 투명도 제거
                img_resized = img.resize((100, 100))
                resized_path = os.path.splitext(abs_image_path)[0] + "_resized" + os.path.splitext(abs_image_path)[1]
                img_resized.save(resized_path)
                abs_image_path = resized_path
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return False
        
        # 셀 주소 검증
        if not re.match("^[A-Z]+[0-9]+$", target_cell):
            logger.error(f"Invalid cell address: {target_cell}")
            return False
        
        try:
            # 이미 열려있는 Excel 애플리케이션 객체 연결
            excel = win32.GetActiveObject("Excel.Application")
            
            # 활성화된 워크북 가져오기
            workbook = excel.ActiveWorkbook
            worksheet = workbook.Worksheets(1)  # 첫 번째 워크시트
            
            # 셀 주소에서 행과 열 추출
            col_letter = re.match("[A-Z]+", target_cell).group()
            row_number = int(re.search("\d+", target_cell).group())
            
            # 셀의 크기를 조정된 이미지의 크기에 맞춤
            worksheet.Cells(row_number, col_letter).ColumnWidth = 100 / 6.25  # 변환 계수
            worksheet.Cells(row_number, col_letter).RowHeight = 100
            
            # 이미지 삽입
            start_cell = worksheet.Range(target_cell)
            left = start_cell.Left
            top = start_cell.Top
            
            worksheet.Shapes.AddPicture(
                abs_image_path, 
                LinkToFile=False, 
                SaveWithDocument=True, 
                Left=left, 
                Top=top, 
                Width=100, 
                Height=100
            )
            
            # 워크북 저장
            workbook.Save()
            logger.info(f"Image inserted successfully to cell {target_cell}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error inserting image to Excel: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in insert_image_to_excel: {str(e)}")
        return False

def process_excel_file(input_path: str) -> Optional[str]:
    """
    엑셀 파일을 처리하는 전체 프로세스를 실행합니다.
    
    Args:
        input_path: 입력 엑셀 파일 경로
        
    Returns:
        Optional[str]: 처리된 파일 경로
    """
    try:
        # 파일 경로 확인
        if not os.path.exists(input_path):
            logger.error(f"파일을 찾을 수 없습니다: {input_path}")
            return None
        
        # 1. XLS -> XLSX 변환 (확장자가 .xls인 경우)
        input_ext = os.path.splitext(input_path)[1].lower()
        input_dir = os.path.dirname(input_path)
        
        if input_ext == '.xls':
            logger.info(f"XLS 파일 감지: {input_path}")
            xlsx_file = convert_xls_to_xlsx(input_dir)
            if xlsx_file:
                logger.info(f"XLS 파일이 XLSX로 변환되었습니다: {xlsx_file}")
                input_path = xlsx_file
            else:
                logger.warning("XLS 파일 변환에 실패했습니다. 원본 파일을 사용합니다.")
        
        # 2. 필요한 컬럼 확인 및 추가
        check_excel_columns(input_path)
        
        # 3. 하이퍼링크 추가
        linked_file = add_hyperlinks_to_excel(input_path)
        
        # 4. 가격 차이 필터링
        filtered_file = filter_excel_by_price_diff(linked_file)
        
        logger.info(f"엑셀 파일 처리가 완료되었습니다: {filtered_file}")
        return filtered_file
        
    except Exception as e:
        logger.error(f"엑셀 파일 처리 중 오류가 발생했습니다: {str(e)}")
        return None 