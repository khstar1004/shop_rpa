#!/usr/bin/env python3
"""
adpanchok.co.kr 스크래퍼 테스트 스크립트
KoryoScraper 클래스의 Selenium 기반 검색 및 제품 정보 추출 기능을 확인합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 고려기프트 스크래퍼 및 관련 모듈 임포트
from core.scraping.koryo_scraper import KoryoScraper
from utils.caching import FileCache

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/koryo_scraper_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# Selenium 관련 로거의 로그 레벨을 높여서 HTML 출력 제한
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('webdriver').setLevel(logging.WARNING)

logger = logging.getLogger('koryo_scraper_test')

def print_product_info(product, detailed=False):
    """제품 정보를 예쁘게 출력하는 헬퍼 함수"""
    print(f"\n{'=' * 50}")
    print(f"제품명: {product.name}")
    print(f"가격: {product.price:,}원")
    print(f"ID: {product.id}")
    print(f"URL: {product.url}")
    print(f"이미지: {product.image_url}")
    
    if detailed:
        print(f"\n제품 코드: {product.product_code}")
        
        if product.specifications:
            print("\n[제품 사양]")
            for key, value in product.specifications.items():
                print(f"  {key}: {value}")
        
        if product.quantity_prices:
            print("\n[수량별 가격]")
            for qty, price in product.quantity_prices.items():
                print(f"  {qty}개: {price:,}원")
        
        if product.image_gallery and len(product.image_gallery) > 1:
            print(f"\n[이미지 갤러리] - {len(product.image_gallery)}개 이미지")
            for i, img in enumerate(product.image_gallery[:3], 1):
                print(f"  {i}. {img}")
            if len(product.image_gallery) > 3:
                print(f"  ... 외 {len(product.image_gallery) - 3}개")
    
    print(f"{'=' * 50}")

def test_search_functionality(scraper):
    """검색 기능 테스트"""
    # 검색어 목록
    search_queries = [
        "볼펜",
        "텀블러",
        "판촉물",
        "기념품",
        "머그컵"
    ]
    
    # 각 검색어로 제품 검색 테스트
    for query in search_queries:
        print(f"\n\n{'#' * 60}")
        print(f"# 검색어: {query}")
        print(f"{'#' * 60}")
        
        # 검색 수행
        logger.info(f"'{query}' 검색 시작")
        products = scraper.search_product(query, max_items=5)
        
        # 결과 처리
        if not products:
            logger.warning(f"'{query}'에 대한 검색 결과가 없습니다.")
            print(f"검색 결과가 없습니다.")
        else:
            logger.info(f"'{query}'에 대한 검색 결과 {len(products)}개 발견")
            print(f"\n총 {len(products)}개 제품 발견:")
            
            # 결과 출력
            for i, product in enumerate(products, 1):
                print(f"\n[제품 {i}/{len(products)}]")
                # 첫 번째 제품은 상세 정보로 출력
                print_product_info(product, detailed=(i==1))
        
        print("\n" + "-" * 80)

def test_category_crawling(scraper):
    """카테고리 크롤링 테스트"""
    print(f"\n\n{'#' * 60}")
    print(f"# 카테고리 목록 가져오기")
    print(f"{'#' * 60}")
    
    try:
        # 카테고리 목록 가져오기
        categories = scraper.get_categories()
        
        if not categories:
            logger.warning("카테고리 목록을 가져오지 못했습니다.")
            print("카테고리를 찾을 수 없습니다.")
            return
        
        logger.info(f"{len(categories)}개 카테고리 발견")
        print(f"\n총 {len(categories)}개 카테고리 발견:")
        
        # 카테고리 목록 출력
        for i, category in enumerate(categories[:10], 1):  # 처음 10개만 표시
            print(f"{i}. {category['name']} (ID: {category['id'][:8]}...)")
        
        if len(categories) > 10:
            print(f"... 외 {len(categories) - 10}개")
        
        # 첫 번째 카테고리의 제품 크롤링 테스트
        if categories:
            selected_category = categories[0]
            print(f"\n\n{'#' * 60}")
            print(f"# 카테고리 '{selected_category['name']}' 제품 크롤링")
            print(f"{'#' * 60}")
            
            logger.info(f"카테고리 '{selected_category['name']}' 크롤링 시작")
            
            # 카테고리 URL 직접 접근
            scraper.driver.get(selected_category['url'])
            
            # 현재 페이지에서 제품 추출
            soup = BeautifulSoup(scraper.driver.page_source, 'lxml')
            product_elements = scraper.extract(soup, scraper.selectors['product_list']['selector'], 
                                              **scraper.selectors['product_list']['options'])
            
            if not product_elements:
                logger.warning(f"카테고리 '{selected_category['name']}'에 제품이 없습니다.")
                print("제품을 찾을 수 없습니다.")
                return
            
            # 첫 번째 제품 상세 정보 추출
            product_data = scraper._extract_list_item(product_elements[0])
            detailed_product = scraper._get_product_details(product_data)
            
            if detailed_product:
                logger.info(f"카테고리 '{selected_category['name']}'에서 제품 추출 성공")
                print(f"\n카테고리 첫 번째 제품 정보:")
                print_product_info(detailed_product, detailed=True)
            else:
                logger.warning(f"카테고리 '{selected_category['name']}'에서 제품 추출 실패")
                print("제품 상세 정보를 가져오지 못했습니다.")
    except Exception as e:
        logger.error(f"카테고리 크롤링 테스트 중 오류 발생: {str(e)}", exc_info=True)
        print(f"카테고리 크롤링 오류: {str(e)}")

def test_direct_category_browsing(scraper):
    """카테고리 직접 브라우징 테스트"""
    print(f"\n\n{'#' * 60}")
    print(f"# 카테고리 직접 브라우징 테스트")
    print(f"{'#' * 60}")
    
    try:
        # 기본 카테고리 사용 (볼펜/사무용품)
        category_id = "013001000"
        category_name = "볼펜/사무용품"
        
        print(f"\n'{category_name}' 카테고리 브라우징 시작")
        logger.info(f"'{category_name}' 카테고리 직접 브라우징 시작")
        
        products = scraper.browse_category(category_id=category_id, max_items=5)
        
        if not products:
            logger.warning(f"'{category_name}' 카테고리에서 제품을 찾지 못했습니다.")
            print(f"카테고리에서 제품을 찾을 수 없습니다.")
        else:
            logger.info(f"카테고리에서 {len(products)}개 제품 발견")
            print(f"\n총 {len(products)}개 제품 발견:")
            
            # 결과 출력
            for i, product in enumerate(products, 1):
                print(f"\n[제품 {i}/{len(products)}]")
                # 첫 번째 제품은 상세 정보로 출력
                print_product_info(product, detailed=(i==1))
    
    except Exception as e:
        logger.error(f"카테고리 직접 브라우징 테스트 중 오류 발생: {str(e)}", exc_info=True)
        print(f"카테고리 브라우징 오류: {str(e)}")

def main():
    logger.info("adpanchok.co.kr 스크래퍼 테스트 시작")
    
    # 캐시 디렉토리 생성 (없을 경우)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 캐시 초기화 - 새로운 캐시 사용 (캐시 비활성화)
    cache = FileCache(cache_dir="cache", duration_seconds=86400, max_size_mb=1024)
    
    try:
        # KoryoScraper 초기화
        scraper = KoryoScraper(max_retries=3, cache=cache, timeout=30, debug=True)
        logger.info("스크래퍼 초기화 완료")
        
        # 검색 기능 테스트
        test_search_functionality(scraper)
        
        # 카테고리 크롤링 테스트
        test_category_crawling(scraper)
        
        # 직접 카테고리 브라우징 테스트 (특정 카테고리 ID 사용)
        test_direct_category_browsing(scraper)
        
        logger.info("테스트 완료")
        print("\n테스트가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}", exc_info=True)
        print(f"\n오류 발생: {str(e)}")
        return 1
    finally:
        # 스크래퍼 종료 (WebDriver 자원 해제)
        if 'scraper' in locals() and hasattr(scraper, 'driver') and scraper.driver:
            try:
                scraper.driver.quit()
                logger.info("WebDriver 자원 해제 완료")
            except Exception as e:
                logger.error(f"WebDriver 자원 해제 중 오류: {str(e)}")
    
    return 0

if __name__ == "__main__":
    # BeautifulSoup 임포트 (test_category_crawling 함수에서 사용)
    from bs4 import BeautifulSoup
    sys.exit(main()) 