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
from core.data_models import Product

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
        
        # 헤더 검사 및 모든 열 이름 출력
        headers = []
        for i, cell in enumerate(sheet[1], 1):
            headers.append(cell.value)
            if cell.value == "메인 이미지":
                formula_col = i
            elif cell.value == "이미지 URL":
                image_col = i
        
        logging.info(f"Found headers: {headers}")
        
        if not image_col or not formula_col:
            logging.warning(f"Image URL column ({image_col}) or Image formula column ({formula_col}) not found")
            return False
            
        # 데이터 행 검사
        count_total = 0
        count_valid = 0
        count_formulas = 0
        
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
                count_formulas += 1
                image_formulas.append(formula_cell.value)
        
        # 결과 출력
        logging.info(f"Found {count_valid}/{count_total} valid image URLs")
        logging.info(f"Found {count_formulas}/{count_total} image formulas")
        
        # URL 및 수식 샘플 출력
        if image_urls:
            for i, url in enumerate(image_urls[:3]):  # 처음 3개만 출력
                logging.info(f"URL sample {i+1}: {url}")
        if image_formulas:
            for i, formula in enumerate(image_formulas[:3]):  # 처음 3개만 출력
                logging.info(f"Formula sample {i+1}: {formula}")
            
        return count_valid > 0
            
    except Exception as e:
        logging.error(f"Error checking Excel file: {str(e)}")
        return False

def extract_product_from_url(scraper, url):
    """URL에서 직접 제품 정보 추출"""
    logging.info(f"Extracting product from URL: {url}")
    
    # URL에서 product_id 추출
    product_id = None
    try:
        if "no=" in url:
            import re
            product_id = re.search(r"no=(\d+)", url).group(1)
    except Exception as e:
        logging.error(f"Error extracting product ID from URL: {str(e)}")
    
    if product_id:
        logging.info(f"Found product ID: {product_id}")
        # ID를 이용한 직접 제품 조회 시도
        product = scraper.get_product(product_id)
        if product:
            return product
    
    # ID로 찾지 못한 경우, 아이템 딕셔너리를 만들어서 상세 정보 추출 시도
    item_data = {
        "link": url,
        "product_id": product_id or f"direct_{datetime.now().timestamp()}",
        "title": "직접 URL 접근",
        "price": 0,
        "image": None
    }
    
    return scraper._get_product_details_sync_playwright(item_data)

def main():
    """고려기프트 직접 URL 테스트 메인 함수"""
    logging.info("=== 고려기프트 직접 URL 테스트 시작 ===")
    
    # 샘플 링크 목록
    sample_links = [
        "https://adpanchok.co.kr/ez/mall.php?cat=013001001&query=view&no=135429",
        "https://adpanchok.co.kr/ez/mall.php?cat=004002000&query=view&no=128440",
        "https://adpanchok.co.kr/ez/mall.php?cat=005008007&query=view&no=134068",
        "https://koreagift.com/ez/mall.php?cat=013001000&query=view&no=136450",
        "https://koreagift.com/ez/mall.php?cat=013002000&query=view&no=133254"
    ]
    
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
    
    # 출력 디렉토리 생성
    os.makedirs("output", exist_ok=True)
    
    # 타임스탬프로 고유한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/koryo_direct_test_{timestamp}.xlsx"
    
    # 각 샘플 링크에서 제품 추출
    products = []
    for url in sample_links:
        product = extract_product_from_url(scraper, url)
        if product:
            products.append(product)
            logging.info(f"Successfully extracted product: {product.name}")
            logging.info(f"  - Price: {product.price}")
            logging.info(f"  - Image URL: {product.image_url}")
            
            # 추가 이미지 갤러리 확인
            if product.image_gallery and len(product.image_gallery) > 0:
                logging.info(f"  - Image gallery: {len(product.image_gallery)} images")
                for i, img_url in enumerate(product.image_gallery[:3]):  # 처음 3개만 출력
                    logging.info(f"    Gallery image {i+1}: {img_url}")
            else:
                logging.info("  - No image gallery found")
        else:
            logging.error(f"Failed to extract product from URL: {url}")
    
    # 검색 기능 테스트
    search_term = "볼펜"
    logging.info(f"\n검색 테스트: '{search_term}'")
    search_products = scraper.search_product(search_term, max_items=5)
    if search_products:
        logging.info(f"Search found {len(search_products)} products for '{search_term}'")
        for i, product in enumerate(search_products):
            logging.info(f"Search result {i+1}: {product.name} (이미지: {product.image_url is not None})")
        
        # 검색 결과도 products 목록에 추가
        products.extend(search_products)
    else:
        logging.warning(f"No search results for '{search_term}'")
    
    # Excel 파일에 저장 테스트
    if products:
        logging.info(f"Saving {len(products)} products to Excel file: {output_file}")
        excel_manager.save_products(products, output_file, "koryo_direct_test")
        
        # Excel 파일 검사
        logging.info(f"Checking original Excel file: {output_file}")
        check_image_urls_in_excel(output_file)
        
        # 후처리 (이미지 포뮬러 수정 등)
        processed_file = excel_manager.post_process_excel_file(output_file)
        logging.info(f"Excel post-processing completed: {processed_file}")
        
        # 이미지 URL 확인
        logging.info("Checking processed Excel file with IMAGE formulas:")
        check_image_urls_in_excel(processed_file)
    else:
        logging.warning("No products to save.")
    
    logging.info("=== 고려기프트 직접 URL 테스트 완료 ===")

if __name__ == "__main__":
    main() 