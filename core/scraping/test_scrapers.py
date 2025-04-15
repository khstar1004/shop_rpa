import logging
import sys
import os
from pathlib import Path
import configparser
import time
import json

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any
from core.scraping.haeoeum_scraper import HaeoeumScraper
from core.scraping.koryo_scraper import KoryoScraper, ScraperConfig
from core.scraping.naver_crawler import NaverShoppingAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'test_scrapers.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 로그 디렉토리 생성
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'test_results'), exist_ok=True)

def load_config() -> configparser.ConfigParser:
    """Load configuration from config.ini"""
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'config.ini')
    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')
    else:
        # Set default values
        config['SCRAPING'] = {
            'max_retries': '3',
            'extraction_timeout': '60000',
            'navigation_timeout': '60000',
            'wait_timeout': '30000',
            'request_delay': '1.0',
            'headless': 'true',
            'debug': 'false',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'cache_ttl': '3600',
            'viewport_width': '1920',
            'viewport_height': '1080',
            'connection_pool_size': '10',
            'ssl_verification': 'true',
            'retry_on_specific_status': '429,503,502,500'
        }
    return config

def test_haeoeum_scraper(url: str) -> Dict[str, Any]:
    """Test Haeoeum scraper with the given URL"""
    logger.info("Testing Haeoeum scraper...")
    config = load_config()
    scraper = HaeoeumScraper(
        max_retries=int(config.get('SCRAPING', 'max_retries', fallback='3')),
        timeout=int(config.get('SCRAPING', 'extraction_timeout', fallback='60000')),
        headless=config.getboolean('SCRAPING', 'headless', fallback=True),
        debug=True  # 디버깅 모드 활성화
    )
    try:
        start_time = time.time()
        # Extract product ID from URL
        product_idx = url.split('p_idx=')[1].split('&')[0]
        logger.info(f"Extracting product with ID: {product_idx}")
        
        # 먼저 이미지 추출 테스트 실행
        logger.info("Running image extraction test...")
        test_results = scraper.test_image_extraction(product_idx)
        logger.info(f"Image test results: Found {len(test_results['combined_result']['images'])} images")
        
        # 전체 상품 정보 추출
        product = scraper.get_product(product_idx)
        elapsed_time = time.time() - start_time
        
        if product:
            # 이미지 갤러리 상태 확인
            if not product.image_gallery:
                logger.warning("Product has no image gallery. Using test results...")
                if test_results['combined_result']['images']:
                    product.image_gallery = test_results['combined_result']['images']
                    logger.info(f"Applied {len(product.image_gallery)} images from test results")
            
            result = {
                'name': product.name,
                'price': product.price,
                'stock_status': product.stock_status or "Unknown",
                'images': product.image_gallery,
                'image_count': len(product.image_gallery),
                'elapsed_time': f"{elapsed_time:.2f}s"
            }
            
            # 결과 저장 (JSON)
            timestamp = int(time.time())
            result_path = os.path.join(project_root, 'test_results', f'haeoeum_{product_idx}_{timestamp}.json')
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {result_path}")
            
            logger.info(f"Success: Extracted product '{product.name}' (₩{product.price}) with {len(product.image_gallery)} images in {elapsed_time:.2f}s")
            return result
        else:
            logger.error(f"Failed: No product extracted in {elapsed_time:.2f}s")
    except Exception as e:
        logger.error(f"Error testing Haeoeum scraper: {str(e)}", exc_info=True)
    return {'error': 'Failed to extract product', 'elapsed_time': 'N/A'}

def test_koryo_scraper(query: str = "우리쌀 라이스칩") -> Dict[str, Any]:
    """Test Koryo scraper by searching for a product and extracting the lowest priced item."""
    logger.info(f"Testing Koryo scraper by searching for '{query}'...")
    config = load_config()
    scraper_config = ScraperConfig(config)
    scraper = KoryoScraper(config=scraper_config)
    try:
        start_time = time.time()
        # Search for the product and get the lowest priced one
        product = scraper.search_and_extract_lowest_price_product(query)
        elapsed_time = time.time() - start_time
        
        if product:
            result = {
                'name': product.name,
                'price': product.price,
                'stock_status': product.stock_status or "Unknown",
                'images': list(product.image_gallery),  # Convert to list for JSON serialization
                'specifications': dict(product.specifications),  # Convert to dict for JSON serialization
                'url': product.url,
                'elapsed_time': f"{elapsed_time:.2f}s"
            }
            logger.info(f"Success: Found lowest price product '{product.name}' (₩{product.price}) in {elapsed_time:.2f}s")
            
            # 결과 저장 (JSON)
            timestamp = int(time.time())
            safe_query = "".join(c for c in query if c.isalnum() or c in [' ', '_']).replace(' ', '_')
            result_path = os.path.join(project_root, 'test_results', f'koryo_{safe_query}_{timestamp}.json')
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {result_path}")
            
            return result
        else:
            logger.error(f"Failed: No product found for query '{query}' in {elapsed_time:.2f}s")
    except Exception as e:
        logger.error(f"Error testing Koryo scraper: {str(e)}", exc_info=True)
    return {'error': f'Failed to find product for query "{query}"', 'elapsed_time': 'N/A'}

def test_naver_scraper(query: str = "우리쌀라이스칩") -> Dict[str, Any]:
    """Test Naver scraper with a search query"""
    logger.info(f"Testing Naver scraper with query '{query}'...")
    config = load_config()
    scraper = NaverShoppingAPI(
        max_retries=int(config.get('SCRAPING', 'max_retries', fallback='3')),
        timeout=int(config.get('SCRAPING', 'extraction_timeout', fallback='60000'))
    )
    try:
        start_time = time.time()
        # Search for the product
        products = scraper.search_product(query, max_items=1)
        elapsed_time = time.time() - start_time
        
        if products:
            product = products[0]
            result = {
                'name': product.name,
                'price': product.price,
                'stock_status': product.stock_status or "Unknown",
                'images': list(product.image_gallery) if product.image_gallery else [],
                'url': product.url,
                'elapsed_time': f"{elapsed_time:.2f}s"
            }
            logger.info(f"Success: Found product '{product.name}' (₩{product.price}) in {elapsed_time:.2f}s")
            return result
        else:
            logger.error(f"Failed: No products found for query '{query}' in {elapsed_time:.2f}s")
    except Exception as e:
        logger.error(f"Error testing Naver scraper: {str(e)}", exc_info=True)
    return {'error': f'Failed to find product for query "{query}"', 'elapsed_time': 'N/A'}

def main():
    logger.info("===== 스크래퍼 테스트 시작 =====")
    # 로그 헤더 출력
    logger.info(f"Python 버전: {sys.version}")
    logger.info(f"프로젝트 경로: {project_root}")
    
    # 환경 체크를 위한 기본 정보 출력
    try:
        import platform
        logger.info(f"OS: {platform.system()} {platform.release()}")
        
        # Playwright 버전 확인
        try:
            from playwright import __version__ as pw_version
            logger.info(f"Playwright 버전: {pw_version}")
        except (ImportError, AttributeError):
            logger.warning("Playwright 버전을 확인할 수 없습니다.")
    except Exception as e:
        logger.warning(f"환경 정보 수집 중 오류: {e}")
    
    # Test URLs and queries
    haeoeum_url = "https://www.jclgift.com/product_w/product_view.asp?p_idx=425978&adCode=A57&adGubun=Z"
    koryo_query = "우리쌀 라이스칩"
    naver_query = "우리쌀라이스칩"

    # Test results
    results = {}

    # Test each scraper
    logger.info("\n=== Testing Haeoeum Scraper ===")
    haeoeum_result = test_haeoeum_scraper(haeoeum_url)
    results['haeoeum'] = haeoeum_result
    logger.info(f"Haeoeum Result: {haeoeum_result}")

    logger.info("\n=== Testing Koryo Scraper ===")
    koryo_result = test_koryo_scraper(koryo_query)
    results['koryo'] = koryo_result
    logger.info(f"Koryo Result: {koryo_result}")

    logger.info("\n=== Testing Naver Scraper ===")
    naver_result = test_naver_scraper(naver_query)
    results['naver'] = naver_result
    logger.info(f"Naver Result: {naver_result}")
    
    # 최종 결과 요약
    logger.info("\n=== 테스트 결과 요약 ===")
    all_success = True
    for name, result in results.items():
        if 'error' in result:
            logger.error(f"{name.capitalize()}: 실패")
            all_success = False
        else:
            logger.info(f"{name.capitalize()}: 성공 - '{result.get('name', 'Unknown')}' (₩{result.get('price', 'Unknown')})")
    
    logger.info(f"최종 결과: {'모든 테스트 성공' if all_success else '일부 테스트 실패'}")
    logger.info("===== 스크래퍼 테스트 종료 =====")

if __name__ == "__main__":
    main() 