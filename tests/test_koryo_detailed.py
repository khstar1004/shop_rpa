#!/usr/bin/env python3
"""
고려기프트 스크래퍼 상세 테스트 스크립트
특정 제품 URL을 사용하여 제품 상세 정보 추출 기능을 집중적으로 테스트합니다.
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
import pprint

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 고려기프트 스크래퍼 및 관련 모듈 임포트
from core.scraping.koryo_scraper import KoryoScraper
from utils.caching import FileCache

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # 디버그 레벨 로깅 활성화
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/koryo_detailed_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger('koryo_detailed_test')

# 테스트할 제품 URL들
TEST_PRODUCT_URLS = [
    # 텀블러 관련 URL
    "https://www.koreagift.co.kr/shop/view.php?index_no=18138",
    # 볼펜 관련 URL
    "https://www.koreagift.co.kr/shop/view.php?index_no=12539",
    # 우산 관련 URL
    "https://www.koreagift.co.kr/shop/view.php?index_no=11786"
]

async def extract_product_details(scraper, url):
    """제품 상세 정보 비동기 추출"""
    item = {
        'link': url,
        'product_id': url.split('=')[-1]  # URL에서 간단하게 ID 추출
    }
    
    # 제품 상세 정보 가져오기
    return await scraper._get_product_details_async(item)

async def run_tests():
    """모든 테스트 실행"""
    logger.info("고려기프트 스크래퍼 상세 테스트 시작")
    
    # 캐시 디렉토리 생성 (없을 경우)
    os.makedirs("cache", exist_ok=True)
    
    # 캐시 초기화 (테스트를 위해 캐시 사용 안 함)
    cache = FileCache(cache_dir="cache", prefix="test_koryo_detailed_")
    
    try:
        # KoryoScraper 초기화
        scraper = KoryoScraper(max_retries=3, cache=cache, timeout=30)
        logger.info("스크래퍼 초기화 완료")
        
        # 각 제품 URL에 대해 테스트
        for url in TEST_PRODUCT_URLS:
            print(f"\n\n{'#' * 60}")
            print(f"# 테스트 URL: {url}")
            print(f"{'#' * 60}")
            
            logger.info(f"제품 상세 정보 추출 시작: {url}")
            
            # 제품 상세 정보 가져오기
            product = await extract_product_details(scraper, url)
            
            if not product:
                logger.error(f"제품 정보를 가져오는데 실패했습니다: {url}")
                print("제품 정보 추출 실패")
                continue
            
            # 제품 정보 출력
            logger.info(f"제품 정보 추출 성공: {product.name}")
            
            print(f"\n[기본 정보]")
            print(f"제품명: {product.name}")
            print(f"가격: {product.price:,}원")
            print(f"ID: {product.id}")
            print(f"URL: {product.url}")
            print(f"제품 코드: {product.product_code or '(없음)'}")
            print(f"이미지 URL: {product.image_url or '(없음)'}")
            
            if product.specifications:
                print(f"\n[제품 사양]")
                for key, value in product.specifications.items():
                    print(f"  {key}: {value}")
            
            if product.quantity_prices:
                print(f"\n[수량별 가격]")
                for qty, price in product.quantity_prices.items():
                    print(f"  {qty}개: {price:,}원")
            
            if product.image_gallery:
                print(f"\n[이미지 갤러리] - {len(product.image_gallery)}개")
                for i, img in enumerate(product.image_gallery[:5], 1):
                    print(f"  {i}. {img}")
                if len(product.image_gallery) > 5:
                    print(f"  ... 외 {len(product.image_gallery) - 5}개")
            
            if product.description:
                desc_preview = product.description[:150] + "..." if len(product.description) > 150 else product.description
                print(f"\n[제품 설명 미리보기]\n{desc_preview}")
            
            # 추출 성공 여부 검증
            success_checks = {
                "제품명": bool(product.name),
                "가격": product.price > 0,
                "URL": bool(product.url),
                "이미지": bool(product.image_url),
                "제품코드": bool(product.product_code),
                "제품설명": bool(product.description),
                "수량가격": bool(product.quantity_prices)
            }
            
            print(f"\n[추출 결과 검증]")
            all_passed = True
            for check_name, passed in success_checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}")
                if not passed:
                    all_passed = False
            
            overall_status = "성공" if all_passed else "일부 필드 누락"
            print(f"\n전체 테스트 결과: {overall_status}")
            
            print("\n" + "-" * 80)
        
        logger.info("모든 테스트 완료")
        print("\n테스트가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}", exc_info=True)
        print(f"\n오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    if sys.platform == 'win32':
        # Windows에서 비동기 이벤트 루프 정책 설정
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 비동기 함수 실행
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code) 