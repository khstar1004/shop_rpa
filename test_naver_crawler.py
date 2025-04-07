#!/usr/bin/env python3
"""
네이버 크롤러 테스트 스크립트
이 스크립트는 NaverShoppingCrawler가 올바르게 작동하는지 확인합니다.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# 로깅 설정 - DEBUG 레벨로 변경
logging.basicConfig(
    level=logging.DEBUG,  # INFO에서 DEBUG로 변경
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("naver_crawler_test")

# 현재 디렉토리를 Python 경로에 추가 (상대 경로 임포트 지원)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.scraping.naver_crawler import NaverShoppingCrawler
    from utils.caching import FileCache
except ImportError as e:
    logger.error(f"필요한 모듈을 가져올 수 없습니다: {e}")
    logger.error("현재 디렉토리에서 실행 중인지 확인하세요.")
    sys.exit(1)

def main():
    """
    테스트 메인 함수
    """
    logger.info("네이버 크롤러 테스트 시작")
    
    # .env 파일 로드
    load_dotenv()
    
    # API 키 확인
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")
    
    if not client_id or not client_secret:
        logger.error(".env 파일에서 API 키를 찾을 수 없습니다.")
        logger.error("client_id와 client_secret이 설정되어 있는지 확인하세요.")
        sys.exit(1)
    
    logger.info(f"API 키 확인: {client_id[:4]}... / {client_secret[:4]}...")
    
    # 캐시 디렉토리 생성
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 캐시 초기화 전에 캐시 비우기
    if os.path.exists(cache_dir):
        logger.info("기존 캐시 파일 삭제 중...")
        for file in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"캐시 파일 삭제 실패: {e}")
    
    # 파일 캐시 초기화
    cache = FileCache(cache_dir=cache_dir, max_size_mb=100)
    
    try:
        # 크롤러 초기화
        crawler = NaverShoppingCrawler(
            max_retries=3,
            cache=cache,
            timeout=30
        )
        
        # 테스트 검색어 설정
        test_query = "텀블러"
        logger.info(f"검색 수행: '{test_query}'")
        
        # 검색 실행
        products = crawler.search_product(test_query, max_items=5)
        
        # 결과 확인
        if products:
            logger.info(f"검색 결과: {len(products)}개 제품 찾음")
            for i, product in enumerate(products[:5], 1):  # 처음 5개 표시
                logger.info(f"제품 {i}: {product.name}")
                logger.info(f"  가격: {product.price}원")
                logger.info(f"  브랜드: {product.brand}")
                logger.info(f"  URL: {product.url}")
                logger.info(f"  이미지: {product.image_url}")
                logger.info(f"  카테고리: {getattr(product, 'category', 'N/A')}")
                logger.info("---")
        else:
            logger.warning(f"'{test_query}'에 대한 검색 결과가 없습니다.")
        
        logger.info("테스트 완료")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 