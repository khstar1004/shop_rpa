#!/usr/bin/env python3
"""
adpanchok.co.kr 스크래퍼 테스트 스크립트
KoryoScraper 클래스의 Playwright 기반 검색 및 제품 정보 추출 기능을 확인합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 고려기프트 스크래퍼 및 관련 모듈 임포트
from core.scraping.koryo_scraper import KoryoScraper, ScraperConfig
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

def test_full_search_flow(scraper):
    """전체 검색 흐름 테스트 (메인페이지 -> 검색 -> 결과 -> 상세정보)"""
    # 다양한 검색어로 테스트
    search_queries = [
        "텀블러",
        "바쏘 텀블러",   # 특정 제품 타겟팅
        "실리콘 텀블러", 
        "스텐 텀블러",
        "락앤락 텀블러"  # 브랜드 포함
    ]
    
    for query in search_queries:
        print(f"\n\n{'#' * 60}")
        print(f"# 전체 검색 흐름 테스트: '{query}'")
        print(f"{'#' * 60}")
        
        # 전체 검색 흐름 실행
        logger.info(f"'{query}'에 대한 전체 검색 흐름 시작")
        
        # 최대 3개 상품만 상세 정보 조회 (시간 단축)
        products = scraper.full_search_flow(query, max_items=3)
        
        if not products:
            logger.warning(f"'{query}'에 대한 검색 결과가 없습니다.")
            print(f"검색 결과가 없습니다.")
            continue
        
        logger.info(f"'{query}'에 대한 전체 검색 완료: {len(products)}개 상품")
        print(f"\n총 {len(products)}개 제품 발견:")
        
        # 결과 출력 및 이미지 확인
        for i, product in enumerate(products, 1):
            print(f"\n[제품 {i}/{len(products)}]")
            detailed = (i==1)  # 첫 번째 제품만 상세 출력
            print_product_info(product, detailed=detailed)
            
            # 이미지 URL 검증
            if product.image_url:
                print(f"이미지 URL 확인: {product.image_url}")
                if "shop_" in product.image_url and product.image_url.endswith(".jpg"):
                    print("✓ 올바른 이미지 URL 형식")
                else:
                    print("⚠ 이미지 URL 형식이 예상과 다름")
            else:
                print("✗ 이미지 URL 없음")
        
        # 결과 저장
        results_file = f"koryo_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        scraper.save_results_to_excel(products, results_file)
        print(f"\n검색 결과를 'output/{results_file}'에 저장했습니다.")
        
        print("\n" + "-" * 80)

def test_direct_product_access(scraper):
    """특정 상품의 직접 접근 테스트"""
    print(f"\n\n{'#' * 60}")
    print(f"# 특정 상품 직접 접근 테스트")
    print(f"{'#' * 60}")
    
    # 실제 상품 ID 목록 (수동으로 확인한 텀블러 ID)
    product_ids = [
        "122793",  # 바쏘 휴대 간편 접이식 실리콘 텀블러 500ml
        "131747",  # 다른 텀블러 ID
        "128939"   # 또 다른 텀블러 ID  
    ]
    
    for product_id in product_ids:
        logger.info(f"상품 ID {product_id} 직접 접근 테스트")
        
        # 상품 정보 조회
        product = scraper.get_product_with_retry(product_id)
        
        if product:
            logger.info(f"상품 ID {product_id} 정보 조회 성공: {product.name}")
            print(f"\n상품 ID {product_id} 정보:")
            print_product_info(product, detailed=True)
            
            # 이미지 URL 검증
            if product.image_url:
                print(f"이미지 URL: {product.image_url}")
                if "shop_" in product.image_url and product.image_url.endswith(".jpg"):
                    print("✓ 올바른 이미지 URL 형식")
                else:
                    print("⚠ 이미지 URL 형식이 예상과 다름")
            else:
                print("✗ 이미지 URL 없음")
            
            # 이미지 갤러리 확인
            if product.image_gallery:
                print(f"\n이미지 갤러리: {len(product.image_gallery)}개")
                for i, img_url in enumerate(product.image_gallery[:3], 1):
                    print(f"  {i}. {img_url}")
                if len(product.image_gallery) > 3:
                    print(f"  ... 외 {len(product.image_gallery) - 3}개")
            else:
                print("\n이미지 갤러리 없음")
        else:
            logger.warning(f"상품 ID {product_id} 정보 조회 실패")
            print(f"\n상품 ID {product_id} 정보를 조회할 수 없습니다.")
        
        print("\n" + "-" * 80)

def test_search_lowest_price_product(scraper):
    """검색 결과에서 가격 낮은순 정렬 후 첫 상품 추출 테스트"""
    print(f"\n\n{'#' * 60}")
    print(f"# 검색 후 최저가 상품 추출 테스트")
    print(f"{'#' * 60}")
    
    # 다양한 검색어로 테스트
    search_queries = [
        "텀블러",           # 일반 검색어
        "스텐 텀블러",      # 재질 특정
        "접이식 텀블러",    # 특징 특정
        "실리콘 텀블러"     # 재질 특정
    ]
    
    for query in search_queries:
        print(f"\n\n{'#' * 60}")
        print(f"# 검색어: '{query}'로 최저가 상품 추출")
        print(f"{'#' * 60}")
        
        # 검색 및 최저가 상품 추출
        logger.info(f"'{query}'로 최저가 상품 추출 시작")
        
        product = scraper.search_and_extract_lowest_price_product(query)
        
        if not product:
            logger.warning(f"'{query}'로 최저가 상품을 찾을 수 없습니다.")
            print(f"최저가 상품을 찾을 수 없습니다.")
            continue
        
        logger.info(f"'{query}'로 최저가 상품 추출 성공: {product.name}")
        print(f"\n최저가 상품 정보:")
        print_product_info(product, detailed=True)
        
        # 이미지 URL 검증
        if product.image_url:
            print(f"\n메인 이미지 URL: {product.image_url}")
            if "shop_" in product.image_url and product.image_url.endswith(".jpg"):
                print("✓ 올바른 이미지 URL 형식")
            else:
                print("⚠ 이미지 URL 형식이 예상과 다름")
        else:
            print("✗ 이미지 URL 없음")
        
        # 이미지 갤러리 검증
        if hasattr(product, 'image_gallery') and product.image_gallery:
            print(f"\n이미지 갤러리: {len(product.image_gallery)}개 이미지")
            for i, img_url in enumerate(product.image_gallery[:3], 1):
                print(f"  {i}. {img_url}")
                
                # 테스트하려는 이미지 URL과 일치 여부 확인
                target_img = "https://adpanchok.co.kr/ez/upload/mall/shop_1680835032763392_0.jpg"
                if img_url == target_img:
                    print(f"  ✅ 찾던 이미지 발견: {img_url}")
            
            if len(product.image_gallery) > 3:
                print(f"  ... 외 {len(product.image_gallery) - 3}개")
        else:
            print("\n이미지 갤러리 없음")
            
        # 수량별 가격 추출 검증
        print("\n[수량별 가격 추출 결과]")
        
        # 기대되는 수량 목록 (높은 수량부터 낮은 수량 순서로)
        expected_quantities = ["5000", "3000", "1000", "500", "300", "200", "100", "50"]
        
        if hasattr(product, 'quantity_prices') and product.quantity_prices:
            print(f"총 {len(product.quantity_prices)}개 수량별 가격 추출됨")
            
            # 수량별 가격 출력 (높은 수량부터)
            sorted_quantities = sorted(product.quantity_prices.keys(), key=lambda x: int(x), reverse=True)
            
            print(f"\n{'수량':^10} | {'가격':^10} | {'비고'}")
            print("-" * 35)
            
            # 기대 수량과 추출된 수량 비교
            found_quantities = []
            missing_quantities = []
            
            for qty in expected_quantities:
                if qty in product.quantity_prices:
                    price = product.quantity_prices[qty]
                    note = "✓"
                    found_quantities.append(qty)
                    print(f"{qty:>10} | {price:>10,.0f} | {note}")
                else:
                    missing_quantities.append(qty)
            
            # 기대된 수량 외의 추가 수량 출력
            for qty in sorted_quantities:
                if qty not in expected_quantities:
                    price = product.quantity_prices[qty]
                    note = "추가 수량"
                    print(f"{qty:>10} | {price:>10,.0f} | {note}")
            
            # 수량 추출 정확도 계산
            accuracy = len(found_quantities) / len(expected_quantities) * 100
            print(f"\n수량 추출 정확도: {accuracy:.1f}% ({len(found_quantities)}/{len(expected_quantities)})")
            
            if missing_quantities:
                print(f"추출되지 않은 수량: {', '.join(missing_quantities)}")
        else:
            print("수량별 가격 정보가 없습니다.")
        
        # 결과 저장
        results_file = f"koryo_lowest_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        scraper.save_results_to_excel([product], results_file)
        print(f"\n검색 결과를 'output/{results_file}'에 저장했습니다.")
        
        print("\n" + "-" * 80)

def main():
    logger.info("adpanchok.co.kr 스크래퍼 테스트 시작")
    
    # 필요한 디렉토리 생성
    os.makedirs("cache", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)
    
    # 캐시 초기화
    cache = FileCache(cache_dir="cache", duration_seconds=86400, max_size_mb=1024)
    
    # 스크래퍼 설정
    config = ScraperConfig(
        max_retries=3,
        timeout=60000,  # 60초
        navigation_timeout=60000,
        wait_timeout=30000,
        request_delay=1.5,
        headless=False,  # GUI 모드로 실행 (디버깅에 유용)
        debug=True
    )
    
    try:
        # KoryoScraper 초기화
        scraper = KoryoScraper(config=config, cache=cache, debug=True)
        logger.info("스크래퍼 초기화 완료")
        
        # 검색 결과에서 가격 낮은순 정렬 후 첫 상품 추출 테스트
        test_search_lowest_price_product(scraper)
        
        # 전체 검색 흐름 테스트 (메인페이지 -> 검색 -> 결과 -> 상세정보)
        # test_full_search_flow(scraper)
        
        # 특정 상품 직접 접근 테스트
        # test_direct_product_access(scraper)
        
        logger.info("테스트 완료")
        print("\n테스트가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}", exc_info=True)
        print(f"\n오류 발생: {str(e)}")
        return 1
    finally:
        # 스크래퍼 종료 (자원 해제)
        if 'scraper' in locals():
            try:
                scraper.close()
                logger.info("스크래퍼 자원 해제 완료")
            except Exception as e:
                logger.error(f"자원 해제 중 오류: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 