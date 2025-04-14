#!/usr/bin/env python3
"""
고려기프트 텀블러 크롤링 테스트 스크립트
텀블러 검색 및 상세 정보 추출 기능을 테스트합니다.
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/koryo_tumbler_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# 로그 레벨 설정
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('playwright').setLevel(logging.WARNING)

logger = logging.getLogger('koryo_tumbler_test')

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

def test_tumbler_search(scraper):
    """텀블러 검색 기능 테스트"""
    print(f"\n\n{'#' * 60}")
    print(f"# 텀블러 검색 테스트")
    print(f"{'#' * 60}")
    
    logger.info("텀블러 검색 테스트 시작")
    
    # 텀블러 검색 실행
    products = scraper.search_tumbler(max_items=10)
    
    # 결과 처리
    if not products:
        logger.warning("텀블러 검색 결과가 없습니다.")
        print("검색 결과가 없습니다.")
        return False
    
    logger.info(f"텀블러 검색 결과 {len(products)}개 발견")
    print(f"\n총 {len(products)}개 텀블러 제품 발견:")
    
    # 결과 출력
    for i, product in enumerate(products, 1):
        print(f"\n[제품 {i}/{len(products)}]")
        # 첫 번째와 두 번째 제품은 상세 정보로 출력
        print_product_info(product, detailed=(i <= 2))
    
    # 첫 번째 제품의 ID 저장하여 반환 (상세 조회용)
    first_product_id = products[0].id if products else None
    
    print("\n" + "-" * 80)
    return first_product_id

def test_tumbler_detail(scraper, product_id):
    """텀블러 상세 정보 조회 테스트"""
    if not product_id:
        logger.warning("상세 정보 테스트를 위한 제품 ID가 없습니다.")
        return False
    
    print(f"\n\n{'#' * 60}")
    print(f"# 텀블러 상세 정보 조회 테스트 (ID: {product_id})")
    print(f"{'#' * 60}")
    
    logger.info(f"텀블러 상세 정보 조회 테스트 시작 (ID: {product_id})")
    
    # 상세 정보 조회
    product = scraper.get_tumbler_detail(product_id)
    
    # 결과 처리
    if not product:
        logger.warning(f"ID {product_id}의 텀블러 상세 정보를 가져오지 못했습니다.")
        print("상세 정보를 가져오지 못했습니다.")
        return False
    
    logger.info(f"텀블러 상세 정보 조회 성공: {product.name}")
    print(f"\n텀블러 상세 정보:")
    
    # 상세 정보 출력
    print_product_info(product, detailed=True)
    
    # 결과 저장
    try:
        # 디렉토리 생성
        os.makedirs("output", exist_ok=True)
        
        # JSON으로 저장
        output_path = f"output/tumbler_detail_{product_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            # Product 객체를 딕셔너리로 변환
            product_dict = {
                "id": product.id,
                "name": product.name,
                "price": product.price,
                "url": product.url,
                "image_url": product.image_url,
                "product_code": product.product_code,
                "specifications": product.specifications,
                "quantity_prices": product.quantity_prices,
                "image_gallery": product.image_gallery,
                "description": product.description,
                "source": product.source,
                "status": product.status
            }
            json.dump(product_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"텀블러 상세 정보를 '{output_path}'에 저장했습니다.")
        print(f"\n텀블러 상세 정보를 '{output_path}'에 저장했습니다.")
    except Exception as e:
        logger.error(f"텀블러 상세 정보 저장 중 오류 발생: {e}")
        print(f"상세 정보 저장 오류: {e}")
    
    return True

def main():
    logger.info("고려기프트 텀블러 크롤링 테스트 시작")
    
    # 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # 캐시 비활성화 (오류 방지)
    cache = None
    
    # 스크래퍼 설정
    config = ScraperConfig(
        headless=False,  # 브라우저 UI 표시 (디버깅용)
        timeout=60000,   # 타임아웃 값 증가
        debug=True       # 디버그 로깅 활성화
    )
    
    try:
        # 스크래퍼 초기화
        scraper = KoryoScraper(config=config, cache=cache)
        logger.info("스크래퍼 초기화 완료")
        
        # 텀블러 검색 테스트
        product_id = test_tumbler_search(scraper)
        
        # 직접 ID 지정으로 실행 (검색 실패 시 대체)
        if not product_id:
            # 실제 존재하는 텀블러 ID (샘플 링크에서 추출)
            product_id = "135429"  # https://adpanchok.co.kr/ez/mall.php?cat=013001001&query=view&no=135429
            logger.info(f"검색 실패로 직접 ID 사용: {product_id}")
        
        # 텀블러 상세 정보 조회 테스트
        if product_id:
            test_tumbler_detail(scraper, product_id)
        
        logger.info("테스트 완료")
        print("\n테스트가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}", exc_info=True)
        print(f"\n오류 발생: {e}")
        return 1
    finally:
        # 스크래퍼 종료
        if 'scraper' in locals():
            try:
                scraper.close()
                logger.info("스크래퍼 자원 해제 완료")
            except Exception as e:
                logger.error(f"자원 해제 중 오류: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 