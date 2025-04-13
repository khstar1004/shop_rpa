#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime
import openpyxl

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 추가 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 라이브러리 로깅 레벨 조정
logging.getLogger('playwright').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# 현재 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.caching import FileCache
from core.scraping.koryo_scraper import KoryoScraper
from core.processing.excel_manager import ExcelManager

def check_image_urls_in_excel(excel_path):
    """엑셀 파일에서 이미지 URL을 확인합니다."""
    logging.info(f"Checking image URLs in: {excel_path}")
    
    try:
        workbook = openpyxl.load_workbook(excel_path)
        sheet = workbook.active
        
        # 헤더 행 건너뛰기
        image_urls = []
        image_formulas = []
        
        # 이미지 URL과 이미지 포뮬러가 있는 열 찾기
        image_col = None
        formula_col = None
        
        # 헤더 검사
        for i, cell in enumerate(sheet[1], 1):
            if cell.value == "메인 이미지":
                formula_col = i
            elif cell.value == "이미지 URL":
                image_col = i
        
        if not image_col or not formula_col:
            logging.warning(f"Image URL column ({image_col}) or Image formula column ({formula_col}) not found")
            return False
            
        # 데이터 행 검사
        count_total = 0
        count_valid = 0
        
        for row in sheet.iter_rows(min_row=2):
            count_total += 1
            
            # 이미지 URL 열 (1-기반 인덱스 -> 0-기반 인덱스)
            url_cell = row[image_col-1]
            formula_cell = row[formula_col-1]
            
            # 이미지 URL 확인
            if url_cell.value and isinstance(url_cell.value, str) and url_cell.value.startswith("http"):
                count_valid += 1
                image_urls.append(url_cell.value)
                
            # 이미지 수식 확인
            if formula_cell.value and isinstance(formula_cell.value, str) and formula_cell.value.startswith("=IMAGE"):
                image_formulas.append(formula_cell.value)
        
        # 결과 출력
        logging.info(f"Found {count_valid}/{count_total} valid image URLs")
        logging.info(f"Found {len(image_formulas)} image formulas")
        
        # 첫 번째 URL 및 수식 샘플 출력
        if image_urls:
            logging.info(f"Sample URL: {image_urls[0]}")
        if image_formulas:
            logging.info(f"Sample formula: {image_formulas[0]}")
            
        return count_valid > 0
            
    except Exception as e:
        logging.error(f"Error checking Excel file: {str(e)}")
        return False

def main():
    """고려기프트 크롤러 테스트 메인 함수"""
    logging.info("=== 고려기프트 크롤러 테스트 시작 ===")
    
    # 캐시 초기화
    cache = FileCache(
        cache_dir="cache",
        duration_seconds=86400,  # 1일
        max_size_mb=1024,  # 1GB
    )
    
    # 스크래퍼 초기화
    scraper = KoryoScraper(cache=cache)
    
    # Excel 매니저 초기화
    excel_manager = ExcelManager({
        "PATHS": {
            "OUTPUT_DIR": "output"
        }
    }, logging.getLogger("excel_manager"))
    
    # 테스트할 검색어 목록
    test_queries = [
        "도자기 머그컵",
        "사무용 볼펜 세트",
        "usb_메모리",
        "티셔츠"
    ]
    
    # 출력 디렉토리 생성
    os.makedirs("output", exist_ok=True)
    
    # 타임스탬프로 고유한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/koryo_test_{timestamp}.xlsx"
    
    # 각 검색어로 테스트
    for query in test_queries:
        logging.info(f"\n검색어 테스트: '{query}'")
        products = scraper.search_product(query, max_items=5)
        
        if not products:
            logging.warning(f"'{query}'에 대한 검색 결과 없음")
            continue
            
        logging.info(f"검색 결과: {len(products)}개 제품")
        
        if products:
            first_product = products[0]
            logging.info(f"첫 번째 제품: {first_product.name}")
            logging.info(f"가격: {first_product.price}")
            logging.info(f"이미지 URL: {first_product.image_url}")
            logging.info(f"제품 URL: {first_product.url}")
            
            # 모든 제품에 이미지 URL이 있는지 확인
            for i, product in enumerate(products):
                if not product.image_url or product.image_url == "":
                    logging.warning(f"제품 {i+1}: {product.name} - 이미지 URL 없음")
                elif not product.image_url.startswith(("http://", "https://")):
                    logging.warning(f"제품 {i+1}: {product.name} - 잘못된 이미지 URL 형식: {product.image_url}")
    
    # Excel 파일에 저장 테스트
    products_all = []
    for query in test_queries:
        products = scraper.search_product(query, max_items=3)
        if products:
            products_all.extend(products)
    
    if products_all:
        logging.info(f"{len(products_all)}개 제품을 Excel 파일에 저장: {output_file}")
        excel_manager.save_products(products_all, output_file, "koryo_test")
        
        # 후처리 (이미지 포뮬러 수정 등)
        processed_file = excel_manager.post_process_excel_file(output_file)
        logging.info(f"Excel 파일 후처리 완료: {processed_file}")
        
        # 이미지 URL 확인
        check_image_urls_in_excel(processed_file)
    else:
        logging.warning("저장할 제품이 없습니다.")
    
    logging.info("=== 고려기프트 크롤러 테스트 완료 ===")

if __name__ == "__main__":
    main() 