"""
Excel 파일 처리를 위한 통합 유틸리티 모듈

Excel 파일 읽기, 쓰기, 변환 및 데이터 처리에 관련된 기능을 제공합니다.
특히 1차 파일과 2차 파일 변환 프로세스에 최적화되어 있습니다.
"""

import logging
import os
import re
import pandas as pd
import numpy as np
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Alignment, Border, PatternFill, Side, Font
from openpyxl.utils import get_column_letter
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

# 기본 로거 설정
logger = logging.getLogger(__name__)

# --- 스타일 상수 정의 ---
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
RED_FONT = Font(name="맑은 고딕", size=10, color="FF0000")

# --- 데이터 변환 유틸리티 ---

def safe_to_numeric(series, errors='coerce'):
    """
    판다스 Series를 숫자로 변환, 일반적인 비숫자 값 처리
    
    Args:
        series: 변환할 Series
        errors: 에러 처리 방법 ('coerce', 'raise', 'ignore')
    
    Returns:
        변환된 Series
    """
    non_numeric_placeholders = ['#REF!', '', '-', '동일상품 없음', '(공백)', '(공백 or 없음)', 
                                '가격이 범위 내에 없거나 검색된 상품이 없음']
    series = series.replace(non_numeric_placeholders, np.nan)
    
    # 쉼표 제거 (천 단위 구분자)
    if series.dtype == 'object':
        series = series.str.replace(',', '', regex=False)
        
    return pd.to_numeric(series, errors=errors)

def clean_url(url):
    """
    URL 정제
    
    Args:
        url: 정제할 URL 문자열
    
    Returns:
        정제된 URL 문자열
    """
    if pd.isna(url) or not url:
        return ""
        
    url_str = str(url).strip()
    
    # 이미지 수식에서 URL 추출 ('=IMAGE("URL", 2)' 형식)
    image_formula_match = re.search(r'=IMAGE\("([^"]+)",\s*\d+\)', url_str)
    if image_formula_match:
        return image_formula_match.group(1)
        
    # http:// 또는 https:// 로 시작하는지 확인
    if url_str and not url_str.startswith(("http://", "https://")):
        url_str = "https://" + url_str
        
    return url_str

# --- Excel 파일 입출력 ---

def read_excel_file(file_path: str) -> pd.DataFrame:
    """
    Excel 파일을 읽어 DataFrame으로 반환
    
    Args:
        file_path: Excel 파일 경로
    
    Returns:
        DataFrame
    """
    try:
        df = pd.read_excel(file_path)
        logger.info(f"파일 읽기 완료: {file_path}, {len(df)}행 로드됨")
        return df
    except Exception as e:
        logger.error(f"파일 읽기 오류: {file_path}, {str(e)}", exc_info=True)
        # 빈 DataFrame 반환 (오류 발생 시)
        return pd.DataFrame()

def save_excel_file(df: pd.DataFrame, file_path: str, apply_formatting: bool = True) -> str:
    """
    DataFrame을 Excel 파일로 저장
    
    Args:
        df: 저장할 DataFrame
        file_path: 저장할 파일 경로
        apply_formatting: 기본 서식 적용 여부
    
    Returns:
        저장된 파일 경로
    """
    try:
        # 출력 디렉토리 확인 및 생성
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 파일 저장
        df.to_excel(file_path, index=False, engine='openpyxl')
        logger.info(f"파일 저장 완료: {file_path}")
        
        # 서식 적용 (선택적)
        if apply_formatting:
            apply_excel_formatting(file_path)
            
        return file_path
    except Exception as e:
        logger.error(f"파일 저장 오류: {file_path}, {str(e)}", exc_info=True)
        return ""

def apply_excel_formatting(file_path: str) -> bool:
    """
    Excel 파일에 기본 서식 적용
    
    Args:
        file_path: 서식을 적용할 Excel 파일 경로
    
    Returns:
        성공 여부
    """
    try:
        wb = load_workbook(file_path)
        ws = wb.active
        
        # 헤더 행 서식 적용
        for cell in ws[1]:
            cell.font = HEADER_FONT
            cell.alignment = CENTER_ALIGNMENT
            cell.border = THIN_BORDER
            cell.fill = GRAY_FILL
            
        # 열 너비 자동 조정 (최소/최대 너비 지정)
        for col in ws.columns:
            max_length = 0
            for cell in col:
                try:
                    if cell.value:
                        cell_length = len(str(cell.value))
                        max_length = max(max_length, cell_length)
                except:
                    pass
            
            col_letter = get_column_letter(col[0].column)
            adjusted_width = min(max(max_length + 2, 8), 50)  # 최소 8, 최대 50 
            ws.column_dimensions[col_letter].width = adjusted_width
            
        # 음수 값에 빨간색 글꼴 적용
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, (int, float)) and cell.value < 0:
                    cell.font = RED_FONT
                    
                # 모든 셀에 테두리 적용
                cell.border = THIN_BORDER
            
        # 파일 저장
        wb.save(file_path)
        logger.info(f"서식 적용 완료: {file_path}")
        return True
    except Exception as e:
        logger.error(f"서식 적용 오류: {file_path}, {str(e)}", exc_info=True)
        return False

# --- 1차 파일 처리 함수 ---

def process_first_stage(input_file: str, output_dir: Optional[str] = None) -> str:
    """
    입력 파일을 읽어 1차 처리 결과 파일 생성
    
    Args:
        input_file: 입력 Excel 파일 경로
        output_dir: 출력 디렉토리 (기본값: 입력 파일 디렉토리 내 'intermediate')
    
    Returns:
        생성된 1차 파일 경로
    """
    start_time = datetime.now()
    logger.info(f"1차 파일 처리 시작: {input_file}")
    
    # 출력 디렉토리 설정
    if not output_dir:
        parent_dir = os.path.dirname(input_file)
        output_dir = os.path.join(parent_dir, "intermediate")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 출력 파일명 설정
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(output_dir, f"{name}_intermediate{ext}")
    
    try:
        # 파일 읽기
        df = read_excel_file(input_file)
        
        if df.empty:
            logger.warning(f"입력 파일이 비어 있음: {input_file}")
            return ""
            
        # 1. 가격차이 컬럼 생성
        if '고려_가격' in df.columns and '네이버_가격' in df.columns:
            df['고려_가격차이'] = df['고려_가격'] - df['네이버_가격']
            df['네이버_가격차이'] = df['네이버_가격'] - df['고려_가격']
            
            # 가격 및 가격차이 숫자형으로 변환 (나누기 전)
            korea_price_numeric = safe_to_numeric(df['고려_가격'])
            naver_price_numeric = safe_to_numeric(df['네이버_가격'])
            korea_diff_numeric = safe_to_numeric(df['고려_가격차이'])
            naver_diff_numeric = safe_to_numeric(df['네이버_가격차이'])

            # 안전한 퍼센트 계산 (0으로 나누기 방지)
            df['고려_가격차이_퍼센트'] = np.divide(korea_diff_numeric * 100, naver_price_numeric, 
                                             out=np.full_like(korea_diff_numeric, np.nan, dtype=np.double), 
                                             where=naver_price_numeric!=0)
            df['네이버_가격차이_퍼센트'] = np.divide(naver_diff_numeric * 100, korea_price_numeric, 
                                              out=np.full_like(naver_diff_numeric, np.nan, dtype=np.double), 
                                              where=korea_price_numeric!=0)
            
            logger.info("가격차이 및 퍼센트 컬럼 생성 완료")
        else:
            logger.warning("가격차이 계산에 필요한 '고려_가격' 또는 '네이버_가격' 컬럼이 없습니다.")
            # 필요한 컬럼 미리 생성 (NaN으로) - apply_second_stage_rules 에서 에러 방지용
            if '고려_가격차이' not in df.columns: df['고려_가격차이'] = np.nan
            if '네이버_가격차이' not in df.columns: df['네이버_가격차이'] = np.nan
            if '고려_가격차이_퍼센트' not in df.columns: df['고려_가격차이_퍼센트'] = np.nan
            if '네이버_가격차이_퍼센트' not in df.columns: df['네이버_가격차이_퍼센트'] = np.nan

        # 2. 기본수량 컬럼 확인
        if '네이버_기본수량' not in df.columns:
            df['네이버_기본수량'] = pd.NA # Use pd.NA for consistency
            logger.info("네이버_기본수량 컬럼 생성 완료")
        
        # 파일 저장
        save_excel_file(df, output_file)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"1차 파일 처리 완료: {output_file} ({duration:.2f}초)")
        
        return output_file
    except Exception as e:
        logger.error(f"1차 파일 처리 오류: {str(e)}", exc_info=True)
        return ""

# --- 2차 파일 처리 함수 ---

def process_second_stage(input_file: str, output_dir: Optional[str] = None) -> str:
    """
    1차 파일을 읽어 2차 처리 결과 파일 생성
    
    Args:
        input_file: 1차 Excel 파일 경로
        output_dir: 출력 디렉토리 (기본값: 입력 파일 디렉토리 내 'final')
    
    Returns:
        생성된 2차 파일 경로
    """
    start_time = datetime.now()
    logger.info(f"2차 파일 처리 시작: {input_file}")
    
    # 출력 디렉토리 설정
    if not output_dir:
        parent_dir = os.path.dirname(input_file)
        output_dir = os.path.join(parent_dir, "final")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 출력 파일명 설정
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    name = name.replace('_intermediate', '').replace('_first_stage', '')
    output_file = os.path.join(output_dir, f"{name}-result{ext}")
    
    try:
        # 파일 읽기
        df = read_excel_file(input_file)
        
        if df.empty:
            logger.warning(f"입력 파일이 비어 있음: {input_file}")
            return ""
            
        # 2차 처리 규칙 적용
        df_result = apply_second_stage_rules(df)
        
        if df_result.empty:
            logger.warning("2차 처리 후 남은 행이 없음")
            return ""
            
        # 파일 저장
        save_excel_file(df_result, output_file)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"2차 파일 처리 완료: {output_file} ({duration:.2f}초)")
        
        return output_file
    except Exception as e:
        logger.error(f"2차 파일 처리 오류: {str(e)}", exc_info=True)
        return ""

def apply_second_stage_rules(df):
    """
    2차 파일 처리 규칙 적용 (data/inputandoutput.txt 기준)
    
    Args:
        df: 1차 파일 DataFrame
        
    Returns:
        처리된 DataFrame
    """
    logger.info("2차 파일 처리 규칙 적용 중...")
    df_processed = df.copy()
    
    # --- 1. 데이터 변환 ---
    # 원본 Excel 파일의 컬럼 이름을 내부 표준 이름으로 변경하고 숫자로 변환
    rename_and_numeric_cols = {
        '판매단가(V포함)': '본사_판매단가',
        '기본수량(1)': '본사_기본수량',
        '판매단가(V포함)(2)': '고려_판매단가',
        '기본수량(2)': '고려_기본수량',
        '판매단가(V포함)(3)': '네이버_판매단가',
        '기본수량(3)': '네이버_기본수량', # 네이버 기본수량은 1차에서 생성될 수도 있음
    }
    
    processed_columns = set(df_processed.columns) # 컬럼 확인 속도 향상

    for original, new in rename_and_numeric_cols.items():
        if original in processed_columns:
            df_processed[new] = safe_to_numeric(df_processed[original])
            # 원본 컬럼과 새 컬럼 이름이 다르고, 새 컬럼이 원본과 다른 이름으로 성공적으로 생성되었으면 원본 삭제 (중복 방지)
            if original != new and new in df_processed.columns:
                 # 원본 컬럼이 numeric_cols의 다른 값으로 사용되지 않는 경우에만 삭제
                 is_original_used_elsewhere = any(val == original for val in rename_and_numeric_cols.values())
                 if not is_original_used_elsewhere:
                     try:
                         df_processed = df_processed.drop(columns=[original])
                         processed_columns.remove(original) # Update processed columns set
                         logger.debug(f"원본 컬럼 '{original}' 삭제됨 (-> '{new}')")
                     except KeyError:
                         logger.warning(f"컬럼 '{original}' 삭제 시도 중 오류 발생 (이미 없음)")

        elif new not in processed_columns: # 원본 컬럼도 없고, 새 이름의 컬럼도 없을 때
            logger.warning(f"원본 컬럼 '{original}'을 찾을 수 없고, 대상 컬럼 '{new}'도 존재하지 않음. '{new}' 컬럼을 NaN으로 생성합니다.")
            df_processed[new] = np.nan
            processed_columns.add(new) # Update processed columns set
        # else: 원본 컬럼은 없지만 새 이름의 컬럼('new')이 이미 존재하는 경우 (e.g., '네이버_기본수량')는 처리하지 않음

    # 1차 처리에서 계산된 컬럼들이 숫자인지 확인 (필수는 아닐 수 있으나 안전 장치)
    calculated_cols_to_check = ['고려_가격차이', '네이버_가격차이', '고려_가격차이_퍼센트', '네이버_가격차이_퍼센트']
    for col in calculated_cols_to_check:
        if col in processed_columns:
            # 이미 numeric 타입이 아니면 변환 시도
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                 df_processed[col] = safe_to_numeric(df_processed[col])
                 logger.debug(f"계산된 컬럼 '{col}'을 숫자 타입으로 변환했습니다.")
        else:
            # 1차 처리에서 필수적으로 생성되어야 하는 컬럼이 없는 경우
            logger.error(f"필수 계산 컬럼 '{col}'이 누락되었습니다. 1차 처리 결과를 확인하세요. 해당 컬럼을 NaN으로 채웁니다.")
            df_processed[col] = np.nan # 오류 방지를 위해 NaN으로 채움
            processed_columns.add(col)

    # 퍼센트 컬럼 형식 확인 및 조정 (100 곱해진 값 -> 0.xx 형식으로)
    percent_cols = ['고려_가격차이_퍼센트', '네이버_가격차이_퍼센트']
    for col in percent_cols:
        if col in df_processed.columns and not df_processed[col].empty:
            sample_value = df_processed[col].dropna().iloc[0] if not df_processed[col].dropna().empty else None
            if sample_value is not None and abs(sample_value) > 1:
                df_processed[col] = df_processed[col] / 100.0
                logger.info(f"퍼센트 형식 조정: {col}")
    
    # --- 2. 음수값(노란색 셀) 필터 적용 ---
    # 고려기프트 또는 네이버 가격차이가 음수인 행만 유지
    yellow_mask = (df_processed['고려_가격차이'] < 0) | (df_processed['네이버_가격차이'] < 0)
    df_processed = df_processed[yellow_mask].copy()
    logger.info(f"노란색 셀 필터 적용 후: {len(df_processed)}행")
    
    # --- 3. 유효성 플래그 초기화 ---
    df_processed['고려_유효'] = df_processed['고려_가격차이'].notna() & (df_processed['고려_가격차이'] < 0)
    df_processed['네이버_유효'] = df_processed['네이버_가격차이'].notna() & (df_processed['네이버_가격차이'] < 0)
    
    # --- 4. 고려기프트 규칙: 가격차이 절대값이 1% 이하면 무효화 ---
    korea_invalid_mask = df_processed['고려_가격차이_퍼센트'].notna() & (df_processed['고려_가격차이_퍼센트'].abs() <= 0.01)
    df_processed.loc[korea_invalid_mask, '고려_유효'] = False
    logger.info(f"고려기프트 가격차이 1% 이하로 무효화된 행: {korea_invalid_mask.sum()}개")
    
    # --- 5. 네이버 규칙: 기본수량 없고 가격차이 절대값이 10% 이하면 무효화 ---
    naver_invalid_mask = (
        df_processed['네이버_기본수량'].isna() & 
        df_processed['네이버_가격차이_퍼센트'].notna() & 
        (df_processed['네이버_가격차이_퍼센트'].abs() <= 0.10)
    )
    df_processed.loc[naver_invalid_mask, '네이버_유효'] = False
    logger.info(f"네이버 기본수량 없고 가격차이 10% 이하로 무효화된 행: {naver_invalid_mask.sum()}개")
    
    # --- 6. 고려기프트와 네이버 모두 무효한 행 제거 ---
    both_valid_mask = df_processed['고려_유효'] | df_processed['네이버_유효']
    df_processed = df_processed[both_valid_mask].copy()
    logger.info(f"유효한 데이터가 있는 행 필터 적용 후: {len(df_processed)}행")
    
    # --- 7. 무효화된 데이터 섹션 지우기 ---
    # 고려기프트 무효 데이터 지우기
    korea_cols = [col for col in df_processed.columns if '(2)' in col or '고려' in col or '고려기프트' in col]
    for col in korea_cols:
        if col in df_processed.columns:
            df_processed.loc[~df_processed['고려_유효'], col] = np.nan
    
    # 네이버 무효 데이터 지우기
    naver_cols = [col for col in df_processed.columns if '(3)' in col or '네이버' in col or col == '공급사명']
    for col in naver_cols:
        if col in df_processed.columns:
            df_processed.loc[~df_processed['네이버_유효'], col] = np.nan
    
    # --- 8. 출력 컬럼 매핑 및 형식 지정 ---
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
    
    # 컬럼 이름 변경
    rename_map = {k: v for k, v in column_mapping.items() if k in df_processed.columns}
    df_result = df_processed.rename(columns=rename_map)
    
    # 이미지 수식에서 URL만 추출
    image_cols = ['해오름(이미지링크)', '고려기프트(이미지링크)', '네이버쇼핑(이미지링크)']
    for col in image_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].astype(str).apply(clean_url)
            # NaN 및 빈 문자열 처리
            df_result[col] = df_result[col].replace(['nan', 'None', ''], np.nan)
    
    # 출력 컬럼 순서 지정
    output_cols = [
        '구분(A/P)', '담당자', '공급사명', '공급처코드', '상품코드', '카테고리(중분류)',
        '상품명', '본사 기본수량', '판매단가1(VAT포함)', '본사링크',
        '고려 기본수량', '판매단가2(VAT포함)', '고려 가격차이', '고려 가격차이(%)', '고려 링크',
        '네이버 기본수량', '판매단가3(VAT포함)', '네이버 가격차이', '네이버가격차이(%)',
        '네이버 공급사명', '네이버 링크',
        '해오름(이미지링크)', '고려기프트(이미지링크)', '네이버쇼핑(이미지링크)'
    ]
    
    # 존재하는 컬럼만 선택
    final_cols = [col for col in output_cols if col in df_result.columns]
    df_result = df_result[final_cols]
    
    # 가격차이 플래그 컬럼 제거
    for col in ['고려_유효', '네이버_유효']:
        if col in df_result.columns:
            df_result = df_result.drop(columns=[col])
    
    logger.info(f"최종 출력 결과: {len(df_result)}행, {len(final_cols)}열")
    return df_result

# --- 전체 프로세스 실행 함수 ---

def run_complete_workflow(input_file: str, output_root_dir: Optional[str] = None, 
                          run_first_stage: bool = True, run_second_stage: bool = True,
                          send_email: bool = False) -> Dict[str, str]:
    """
    전체 처리 워크플로우 실행 (1차 및 2차 파일 처리)
    
    Args:
        input_file: 입력 Excel 파일 경로
        output_root_dir: 출력 루트 디렉토리 (기본값: 입력 파일과 동일한 디렉토리)
        run_first_stage: 1차 처리 실행 여부
        run_second_stage: 2차 처리 실행 여부
        send_email: 이메일 전송 여부
    
    Returns:
        처리 결과 정보 dictionary
    """
    start_time = datetime.now()
    logger.info(f"전체 워크플로우 시작: {input_file}")
    
    results = {
        "input_file": input_file,
        "first_stage_file": None,
        "second_stage_file": None,
        "email_sent": False,
        "start_time": start_time.isoformat(),
        "status": "processing"
    }
    
    try:
        # 출력 디렉토리 설정
        if not output_root_dir:
            output_root_dir = os.path.dirname(input_file)
            
        first_stage_dir = os.path.join(output_root_dir, "intermediate")
        second_stage_dir = os.path.join(output_root_dir, "final")
        
        # 1차 파일 처리
        if run_first_stage:
            first_stage_file = process_first_stage(input_file, first_stage_dir)
            results["first_stage_file"] = first_stage_file
            
            if not first_stage_file:
                error_msg = "1차 파일 처리 실패: intermediate 파일이 생성되지 않았습니다."
                logger.error(error_msg)
                results["status"] = "error"
                results["error"] = error_msg
                print(f"\n[오류] {error_msg}")
                return results
        else:
            # 1차 처리 생략 시 기존 파일 경로 추정
            base_name = os.path.basename(input_file)
            name, ext = os.path.splitext(base_name)
            first_stage_file = os.path.join(first_stage_dir, f"{name}_intermediate{ext}")
            
            if not os.path.exists(first_stage_file):
                error_msg = f"1차 파일이 존재하지 않음: {first_stage_file}"
                logger.error(error_msg)
                results["status"] = "error"
                results["error"] = error_msg
                print(f"\n[오류] {error_msg}")
                return results
                
            results["first_stage_file"] = first_stage_file
        
        # 이메일 전송 (필요시)
        if send_email and results["first_stage_file"]:
            try:
                # 이메일 전송 로직은 utils/preprocessing.py의 send_report_email 함수 호출
                from utils.preprocessing import send_report_email
                email_sent = send_report_email(results["first_stage_file"], recipient_email='dasomas@kakao.com')
                results["email_sent"] = email_sent
                logger.info(f"이메일 전송 결과: {'성공' if email_sent else '실패'}")
            except Exception as e:
                logger.error(f"이메일 전송 중 오류: {str(e)}", exc_info=True)
                results["email_sent"] = False
        
        # 2차 파일 처리
        if run_second_stage and results["first_stage_file"]:
            second_stage_file = process_second_stage(results["first_stage_file"], second_stage_dir)
            results["second_stage_file"] = second_stage_file
            
            if not second_stage_file:
                error_msg = "2차 파일 처리 실패: final 파일이 생성되지 않았습니다."
                logger.error(error_msg)
                results["status"] = "error"
                results["error"] = error_msg
                print(f"\n[오류] {error_msg}")
                return results
            else:
                results["status"] = "success"
                print(f"\n[성공] 최종 보고서가 생성되었습니다: {second_stage_file}")
        elif not run_second_stage:
            results["status"] = "success"
        
        # 처리 시간 기록
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = duration
        
        logger.info(f"전체 워크플로우 완료: {duration:.2f}초 소요")
        return results
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        error_msg = f"워크플로우 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        results["status"] = "error"
        results["error"] = str(e)
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = duration
        
        print(f"\n[오류] {error_msg}")
        return results

# 직접 실행 시 테스트 코드
if __name__ == "__main__":
    import sys
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
        if len(sys.argv) > 2 and sys.argv[2] == "second_only":
            # 2차 처리만 실행
            process_second_stage(input_file)
        else:
            # 전체 워크플로우 실행
            results = run_complete_workflow(input_file)
            print(f"처리 결과: {results['status']}")
            if results['first_stage_file']:
                print(f"1차 파일: {results['first_stage_file']}")
            if results['second_stage_file']:
                print(f"2차 파일: {results['second_stage_file']}")
    else:
        print("사용법: python excel_processor.py <입력파일경로> [second_only]") 