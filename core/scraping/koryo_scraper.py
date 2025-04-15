"""
고려기프트 웹사이트 스크래퍼 모듈 (Playwright 동기 API 기반)
"""

import hashlib
import logging
import os
import re
import time
import urllib.parse
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, ContextManager, Generator
from urllib.parse import urljoin
from dataclasses import dataclass, field
import configparser
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from playwright.sync_api import sync_playwright, Playwright, Browser, Page, Locator, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

# Add imports for caching
from utils.caching import FileCache, cache_result

from core.data_models import Product, ProductStatus
from core.scraping.base_multi_layer_scraper import BaseMultiLayerScraper
from core.scraping.extraction_strategy import ExtractionStrategy, strategies
from core.scraping.selectors import KORYO_SELECTORS

# Add imports for threading
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ScraperConfig:
    """스크래퍼 설정을 위한 데이터 클래스"""
    def __init__(self, config: configparser.ConfigParser):
        self.max_retries = int(config.get('SCRAPING', 'max_retries', fallback='3'))
        self.timeout = int(config.get('SCRAPING', 'extraction_timeout', fallback='60000'))
        self.navigation_timeout = int(config.get('SCRAPING', 'navigation_timeout', fallback='60000'))
        self.wait_timeout = int(config.get('SCRAPING', 'wait_timeout', fallback='30000'))
        self.request_delay = float(config.get('SCRAPING', 'request_delay', fallback='1.0'))
        self.headless = config.getboolean('SCRAPING', 'headless', fallback=True)
        self.debug = config.getboolean('SCRAPING', 'debug', fallback=False)
        self.user_agent = config.get('SCRAPING', 'user_agent', fallback='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36')
        self.cache_ttl = int(config.get('SCRAPING', 'cache_ttl', fallback='3600'))
        self.viewport_width = int(config.get('SCRAPING', 'viewport_width', fallback='1920'))
        self.viewport_height = int(config.get('SCRAPING', 'viewport_height', fallback='1080'))
        self.connection_pool_size = int(config.get('SCRAPING', 'connection_pool_size', fallback='10'))
        self.ssl_verification = config.getboolean('SCRAPING', 'ssl_verification', fallback=True)
        self.retry_on_specific_status = [int(x) for x in config.get('SCRAPING', 'retry_on_specific_status', fallback='429,503,502,500').split(',')]
        self.base_url = config.get('KORYO', 'base_url', fallback='https://adpanchok.co.kr')


class KoryoScraper(BaseMultiLayerScraper):
    """
    고려기프트 스크래퍼 - Playwright 동기 API 활용

    특징:
    - Playwright 기반 웹 브라우저 자동화
    - DOM 및 텍스트 기반 추출 전략
    - 명시적 대기 사용
    - 캐싱 지원
    """

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        cache: Optional[FileCache] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        connect_timeout: Optional[int] = None,
        read_timeout: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        debug: bool = False,
        user_agent: Optional[str] = None
    ):
        # 설정 초기화
        self.config = config or ScraperConfig(self._load_config())
        
        # 기본값 설정
        self.max_retries = max_retries or self.config.max_retries
        self.timeout = timeout or self.config.timeout
        self.connect_timeout = connect_timeout or self.config.navigation_timeout
        self.read_timeout = read_timeout or self.config.wait_timeout
        self.cache_ttl = cache_ttl or self.config.cache_ttl
        self.debug = debug
        
        # user_agent 설정 (새로 추가)
        if user_agent:
            self.config.user_agent = user_agent
        
        # 캐시 초기화
        self.cache = cache or FileCache(
            cache_dir="cache",  # 기본 캐시 디렉토리
            duration_seconds=self.cache_ttl,  # 캐시 유효 기간
            max_size_mb=1024,  # 최대 캐시 크기 1GB
            enable_compression=True  # 압축 활성화
        )
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # 기본 URL 설정
        self.base_url = self.config.base_url
        
        # Thread-local storage 초기화
        self._thread_local = threading.local()
        self._thread_local.playwright = None
        self._thread_local.browser = None
        self._thread_local.context = None
        self._thread_local.page = None

    @contextmanager
    def thread_context(self) -> Generator[Tuple[Optional[Browser], Optional[Page]], None, None]:
        """Playwright 브라우저 컨텍스트를 관리하는 컨텍스트 매니저"""
        try:
            # Playwright 인스턴스 생성
            if not hasattr(self._thread_local, 'playwright') or not self._thread_local.playwright:
                self._thread_local.playwright = sync_playwright().start()
            
            # 브라우저 실행 옵션 설정
            launch_options = {
                "headless": self.config.headless,
                "args": ["--disable-dev-shm-usage", "--no-sandbox"],
                "ignore_default_args": ["--enable-automation"],
            }
            
            # SSL 검증 설정
            if not self.config.ssl_verification:
                launch_options["ignore_https_errors"] = True
            
            # 브라우저 실행
            if not hasattr(self._thread_local, 'browser') or not self._thread_local.browser:
                self._thread_local.browser = self._thread_local.playwright.chromium.launch(**launch_options)
            
            # 컨텍스트 생성
            context_options = {
                "viewport": {"width": self.config.viewport_width, "height": self.config.viewport_height},
                "user_agent": self.config.user_agent,
                "ignore_https_errors": not self.config.ssl_verification,
                "java_script_enabled": True,
            }
            
            if not hasattr(self._thread_local, 'context') or not self._thread_local.context:
                self._thread_local.context = self._thread_local.browser.new_context(**context_options)
            
            # 페이지 생성 및 타임아웃 설정
            if not hasattr(self._thread_local, 'page') or not self._thread_local.page:
                self._thread_local.page = self._thread_local.context.new_page()
                self._thread_local.page.set_default_timeout(self.timeout)
                self._thread_local.page.set_default_navigation_timeout(self.timeout)
            
            yield self._thread_local.browser, self._thread_local.page
            
        except Exception as e:
            self.logger.error(f"브라우저 컨텍스트 생성 중 오류 발생: {e}", exc_info=True)
            yield None, None
        finally:
            # 리소스 정리
            try:
                if hasattr(self._thread_local, 'page') and self._thread_local.page:
                    self._thread_local.page.close()
                    self._thread_local.page = None
                
                if hasattr(self._thread_local, 'context') and self._thread_local.context:
                    self._thread_local.context.close()
                    self._thread_local.context = None
                
                if hasattr(self._thread_local, 'browser') and self._thread_local.browser:
                    self._thread_local.browser.close()
                    self._thread_local.browser = None
                
                if hasattr(self._thread_local, 'playwright') and self._thread_local.playwright:
                    self._thread_local.playwright.stop()
                    self._thread_local.playwright = None
            except Exception as e:
                self.logger.error(f"리소스 정리 중 오류 발생: {e}", exc_info=True)

    async def search_product(self, query: str, max_items: int = 50) -> List[Product]:
        """제품 검색 수행 (비동기)"""
        if not query:
            return []
            
        # 작업메뉴얼 기준 상품명 전처리
        query = self._preprocess_product_name(query)
        self.logger.info(f"전처리된 검색어: '{query}'")
            
        encoded_query = urllib.parse.quote(query)
        search_url = f"{self.base_url}/ez/mall.php?search_str={encoded_query}"
        
        self.logger.info(f"검색 시작: {query} (URL: {search_url})")
        
        async with self as scraper:
            try:
                response = await scraper._thread_local.page.goto(search_url, wait_until='domcontentloaded')
                if not response or not response.ok:
                    return [self._create_no_match_product(query, is_error=True)]
                
                # 검색 결과 추출
                products = await self._extract_search_results(scraper._thread_local.page, max_items)
                if not products:
                    return [self._create_no_match_product(query)]
                    
                return products
                
            except Exception as e:
                self.logger.error(f"검색 중 오류 발생: {e}", exc_info=True)
                return [self._create_no_match_product(query, is_error=True)]

    async def _extract_search_results(self, page, max_items: int = 50) -> List[Product]:
        """검색 결과 페이지에서 제품 정보 추출 (비동기)"""
        products = []
        try:
            # 1. 정확한 상품 목록 컨테이너 찾기
            container_selectors = [
                "ul.prd_list",  # 가장 공통적인 컨테이너
                "div.prd_list_wrap",
                "table.mall_list",
                ".best100_tab",
                "div.product_list"
            ]
            
            result_container = None
            for selector in container_selectors:
                container = page.locator(selector).first
                if await container.count() > 0 and await container.is_visible(timeout=1000):
                    result_container = container
                    self.logger.info(f"검색 결과 컨테이너 찾음: {selector}")
                    break
            
            # 컨테이너를 찾지 못하면 전체 페이지에서 검색
            if not result_container:
                result_container = page.locator("body")
                self.logger.info("특정 컨테이너를 찾지 못해 전체 페이지에서 검색")
            
            # 2. 제품 항목 찾기
            item_selectors = [
                "li.prd",  # 메인 목록 형식
                "ul.prd_list li",  # 일반 목록
                "table.mall_list td.prd",  # 테이블 형식
                ".product",  # best100 등에서 사용
                ".prd-list .prd-item"  # 대체 형식
            ]
            
            # 상품 목록 찾기
            items = None
            items_count = 0
            
            for selector in item_selectors:
                locator = result_container.locator(selector)
                count = await locator.count()
                
                if count > 0:
                    items = locator
                    items_count = count
                    self.logger.info(f"제품 항목 {count}개 찾음: {selector}")
                    break
            
            if not items or items_count == 0:
                self.logger.warning("제품 항목을 찾지 못함")
                return []
            
            # 3. 각 상품 정보 추출
            for i in range(min(items_count, max_items)):
                try:
                    item = items.nth(i)
                    
                    # 3.1. 상품명
                    title = None
                    title_selectors = [".name", "div.name", "p.name", "h3.name", ".goods_name", ".product_name"]
                    
                    for selector in title_selectors:
                        title_elem = item.locator(selector).first
                        if await title_elem.is_visible(timeout=500):
                            title = await title_elem.text_content(timeout=1000)
                            if title and len(title.strip()) > 0:
                                title = title.strip()
                                break
                    
                    if not title or len(title.strip()) < 2:
                        self.logger.debug(f"항목 #{i}: 제목 없음, 건너뜀")
                        continue
                    
                    # 3.2. 상품 링크 및 ID
                    link = None
                    product_id = None
                    
                    if await item.get_attribute("href"):
                        link = await item.get_attribute("href")
                    else:
                        link_selectors = ["a", "div.name > a", "p.name > a", ".img > a"]
                        for selector in link_selectors:
                            link_elem = item.locator(selector).first
                            if await link_elem.count() > 0:
                                href = await link_elem.get_attribute("href")
                                if href and ("mall.php" in href or "goods_view" in href):
                                    link = href
                                    break
                    
                    if not link:
                        self.logger.debug(f"항목 #{i}: 링크 없음, 건너뜀")
                        continue
                    
                    no_match = re.search(r'no=(\d+)', link)
                    if no_match:
                        product_id = no_match.group(1)
                    
                    if not link.startswith(("http://", "https://")):
                        link = urljoin(self.base_url, link)
                    
                    # 3.3. 가격 정보
                    price = 0
                    price_selectors = [".price", "div.price", "p.price", "strong.price", "span.price"]
                    
                    for selector in price_selectors:
                        price_elem = item.locator(selector).first
                        if await price_elem.is_visible(timeout=500):
                            price_text = await price_elem.text_content(timeout=1000)
                            if price_text:
                                price_match = re.search(r'[\d,]+', price_text)
                                if price_match:
                                    try:
                                        price = float(price_match.group().replace(",", ""))
                                        break
                                    except ValueError:
                                        pass
                    
                    # 3.4. 이미지 URL
                    image_url = None
                    img_selectors = [".img img", "div.pic img", ".thumb img", "img.prd_img"]
                    
                    for selector in img_selectors:
                        img_elem = item.locator(selector).first
                        if await img_elem.is_visible(timeout=500):
                            src = await img_elem.get_attribute("src")
                            if src:
                                if not src.startswith(("http://", "https://")):
                                    src = urljoin(self.base_url, src)
                                image_url = src
                                break
                    
                    # 3.5. 제품 데이터 생성
                    if not product_id:
                        product_id = hashlib.md5(link.encode()).hexdigest()
                    
                    product = Product(
                        id=product_id,
                        name=title,
                        price=price,
                        url=link,
                        image_url=image_url,
                        status="OK" if price > 0 else "Price Not Found"
                    )
                    
                    products.append(product)
                    self.logger.debug(f"항목 #{i}: '{title}' 추출 성공")
                    
                except Exception as item_error:
                    self.logger.error(f"항목 #{i} 처리 오류: {item_error}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"검색 결과 추출 오류: {e}", exc_info=True)
        
        self.logger.info(f"최종 추출된 제품 수: {len(products)}")
        return products

    def _preprocess_product_name(self, product_name: str) -> str:
        """
        작업메뉴얼 기준 상품명 전처리
        1) 상품명에 '1-'가 포함된 경우 - '1-' 및 앞의 숫자/하이픈 제거
        2) 상품명에 '/' 이나 '(번호)' 등이 들어있는 경우 - 해당 특수문자, 괄호 등 제거
        """
        if not product_name:
            return product_name
            
        # 1) "1-" 패턴 및 숫자-하이픈 제거
        processed_name = re.sub(r'^\d+-', '', product_name)
        processed_name = re.sub(r'^\d+\s*-\s*', '', processed_name)
        
        # 2) 특수문자 및 괄호 처리
        processed_name = re.sub(r'\([0-9]+\)', '', processed_name)  # (숫자) 제거
        processed_name = processed_name.replace('/', ' ')  # 슬래시 제거
        
        # 공백 정리
        processed_name = re.sub(r'\s+', ' ', processed_name).strip()
        
        if processed_name != product_name:
            self.logger.info(f"상품명 전처리: '{product_name}' -> '{processed_name}'")
            
        return processed_name

    def _create_no_match_product(self, query: str, is_error: bool = False) -> Product:
        """검색 결과 없거나 오류 발생 시 기본 제품 생성"""
        prefix = "검색 오류" if is_error else "검색 결과 없음"
        
        # 현재 타임스탬프로 고유한 ID 생성
        timestamp = int(datetime.now().timestamp())
        product_id = f"no_match_{timestamp}"
        
        # 실제 존재하는 고려기프트 기본 이미지 사용
        # 실제 고려기프트 사이트에는 이러한 기본 이미지가 존재함
        image_url = f"{self.base_url}/img/ico_shop.gif"
        
        return Product(
            id=product_id,
            name=f"{prefix} - {query}",
            source="koryo",
            price=0,
            url="",
            image_url=image_url,
            status="Extraction Failed"
        )
        
    def _validate_and_normalize_product(self, product: Product) -> bool:
        """
        제품 데이터 유효성 검증 및 이미지 URL 표준화
        
        Args:
            product: 검증할 제품 객체
            
        Returns:
            bool: 유효한 제품인지 여부
        """
        if not product:
            return False
            
        # 1. 필수 필드 검증
        required_fields = {
            'name': product.name,
            'url': product.url,
            'source': product.source
        }
        
        for field, value in required_fields.items():
            if not value:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        # 2. 이미지 URL 처리
        if not product.image_url or product.image_url == "https://koreagift.co.kr/img/ico_shop.gif":
            # 기본 이미지를 사용하지 않고 상품 ID를 활용한 실제 이미지 URL 패턴 적용
            if hasattr(product, 'id') and product.id:
                # 상품 ID가 있는 경우 해당 ID를 활용한 예상 이미지 URL 생성
                product_id = product.id
                if "adpanchok.co.kr" in product.url:
                    # 예: https://adpanchok.co.kr/ez/upload/mall/shop_상품ID_0.jpg
                    product.image_url = f"https://adpanchok.co.kr/ez/upload/mall/shop_{product_id}_0.jpg"
                else:
                    # 예: https://koreagift.co.kr/ez/upload/mall/shop_상품ID_0.jpg
                    product.image_url = f"https://koreagift.co.kr/ez/upload/mall/shop_{product_id}_0.jpg"
                self.logger.info(f"상품 '{product.name}'의 이미지 URL 없음, ID 기반 URL 생성: {product.image_url}")
            else:
                # ID를 사용할 수 없는 경우 기본 이미지 사용
                product.image_url = "https://koreagift.co.kr/img/ico_shop.gif"  # 기본 이미지 설정
                self.logger.warning(f"상품 '{product.name}'의 이미지 URL이 없어 기본 이미지를 사용합니다.")
        else:
            # 2.1 상대 경로를 절대 경로로 변환
            if not product.image_url.startswith(('http://', 'https://', '//')):
                if product.image_url.startswith('/'):
                    # 루트 상대 경로
                    if "adpanchok.co.kr" in product.url:
                        product.image_url = f"https://adpanchok.co.kr{product.image_url}"
                    else:
                        product.image_url = f"https://koreagift.co.kr{product.image_url}"
                else:
                    # 상대 경로
                    product.image_url = urljoin(product.url, product.image_url)
                    
            # 2.2 프로토콜 표준화
            if product.image_url.startswith('http:'):
                product.image_url = 'https:' + product.image_url[5:]
            elif product.image_url.startswith('//'):
                product.image_url = 'https:' + product.image_url
        
        # 3. 이미지 갤러리 처리
        if product.image_gallery:
            normalized_gallery = []
            for img_url in product.image_gallery:
                if not img_url:
                    continue
                    
                # 3.1 상대 경로를 절대 경로로 변환
                if not img_url.startswith(('http://', 'https://', '//')):
                    if img_url.startswith('/'):
                        # 루트 상대 경로
                        if "adpanchok.co.kr" in product.url:
                            img_url = f"https://adpanchok.co.kr{img_url}"
                        else:
                            img_url = f"https://koreagift.co.kr{img_url}"
                    else:
                        # 상대 경로
                        img_url = urljoin(product.url, img_url)
                        
                # 3.2 프로토콜 표준화
                if img_url.startswith('http:'):
                    img_url = 'https:' + img_url[5:]
                elif img_url.startswith('//'):
                    img_url = 'https:' + img_url
                    
                normalized_gallery.append(img_url)
                
            product.image_gallery = normalized_gallery
        
        return True

    def _extract_list_item(self, element_locator: Locator) -> Optional[Dict]:
        """리스트 아이템에서 제품 정보 추출"""
        try:
            # 기본 정보 추출
            title = self._safe_get_text(element_locator.locator('.name'))
            if not title:
                return None
                
            # 가격 추출
            price_text = self._safe_get_text(element_locator.locator('.price'))
            price = 0
            if price_text:
                price_match = re.search(r'[\d,]+', price_text)
                if price_match:
                    try:
                        price = float(price_match.group().replace(",", ""))
                    except ValueError:
                        pass
                        
            # 링크 추출
            link = self._safe_get_attribute(element_locator.locator('a'), 'href')
            if not link:
                return None
                
            # 상품 ID 추출
            product_id = None
            no_match = re.search(r'no=(\d+)', link)
            if no_match:
                product_id = no_match.group(1)
            else:
                product_id = hashlib.md5(link.encode()).hexdigest()
                
            # 이미지 URL 추출
            image_url = self._safe_get_attribute(element_locator.locator('img'), 'src')
            
            # 상대 URL 처리
            if link and not link.startswith(('http://', 'https://')):
                link = urljoin(self.base_url, link)
                
            if image_url and not image_url.startswith(('http://', 'https://')):
                image_url = urljoin(self.base_url, image_url)
                
            # 제품 객체 생성
            product = Product(
                id=product_id,
                name=title,
                price=price,
                url=link,
                image_url=image_url,
                status="OK" if price > 0 else "Price Not Found"
            )
            
            return product
            
        except Exception as e:
            self.logger.error(f"리스트 아이템 추출 중 오류: {e}", exc_info=True)
            return None

    def _extract_product_details(self, page: Page, item: Dict) -> Optional[Product]:
        """상세 페이지에서 제품 정보 추출 (Refined based on koryoproductpage.html)"""
        product_id = item.get("product_id")
        product_link = item.get("link")
        
        if not product_link:
            self.logger.warning("상품 링크 정보 없음")
            return None
            
        if not product_id:
            product_id = hashlib.md5(product_link.encode()).hexdigest()
        
        # 디버깅: 페이지 HTML 구조 로깅
        try:
            page_title = page.title()
            self.logger.debug(f"상세 페이지 제목: {page_title}")
            
            # 주요 셀렉터 존재 여부 확인
            selectors_to_check = [
                '.product_name', 
                '#main_price', 
                '.product_picture', 
                '.prd_detail',
                'table.tbl_info'
            ]
            
            for selector in selectors_to_check:
                try:
                    count = page.locator(selector).count()
                    is_visible = False
                    if count > 0:
                        try:
                            is_visible = page.locator(selector).first.is_visible(timeout=1000)
                        except Exception:
                            pass
                    self.logger.debug(f"셀렉터 '{selector}': 개수={count}, 보임={is_visible}")
                except Exception as e:
                    self.logger.debug(f"셀렉터 '{selector}' 확인 중 오류: {e}")
        except Exception as debug_error:
            self.logger.warning(f"페이지 디버깅 정보 수집 중 오류: {debug_error}")
        
        # koryoproductpage.html 분석 기반 정확한 셀렉터 사용
        try:
            # 1. 상품명 추출 - koryoproductpage.html .product_name
            title = item.get("title", "Unknown Product") # Use list title as fallback
            
            # 페이지 안정화를 위해 잠시 대기
            page.wait_for_timeout(2000)
            
            # 상품명 추출 시도 (다양한 셀렉터)
            title_selectors = [
                '.product_name',
                '.goods_name',
                'h3.name',
                '.detail_title'
            ]
            
            title_found = False
            for selector in title_selectors:
                title_locator = page.locator(selector).first
                try:
                    if title_locator.count() > 0 and title_locator.is_visible(timeout=5000):
                        detail_title = title_locator.text_content(timeout=3000)
                        if detail_title and len(detail_title.strip()) > 1:
                            title = detail_title.strip()
                            self.logger.debug(f"상세 페이지 상품명 확인 (셀렉터: {selector}): {title}")
                            title_found = True
                            break
                except Exception as title_error:
                    self.logger.debug(f"셀렉터 '{selector}'로 상품명 추출 중 오류: {title_error}")
            
            if not title_found:
                self.logger.warning("모든 셀렉터로 상품명 추출 실패, 목록 페이지 값 사용")
                title = item.get("title", "Unknown Product")
            
            # 2. 상품 코드 추출 - koryoproductpage.html 텍스트 포함 div/span/td
            product_code = None
            code_locators = [
                page.locator('//div[contains(text(), "상품코드")]').first,
                page.locator('//span[contains(text(), "상품코드")]').first,
                page.locator('//td[contains(text(), "상품코드")]').first
            ]
            for locator in code_locators:
                if locator.count() > 0 and locator.is_visible(timeout=3000):
                    code_text = locator.text_content(timeout=2000)
                    if code_text:
                        code_match = re.search(r'상품코드\s*[:\-]?\s*([A-Za-z0-9_-]+)', code_text)
                        if code_match:
                            product_code = code_match.group(1).strip()
                            self.logger.debug(f"상품 코드 찾음: {product_code}")
                            break
            
            # 3. 가격 정보 추출 - koryoproductpage.html #main_price
            price = item.get("price", 0.0) # Use list price as fallback
            price_locator = page.locator("#main_price").first
            if price_locator.is_visible(timeout=5000): # Increased timeout
                price_text = price_locator.text_content(timeout=3000)
                if price_text:
                    price_match = re.search(r'[\d,]+', price_text)
                    if price_match:
                        try:
                            detail_price = float(price_match.group().replace(",", ""))
                            if detail_price > 0: # Use detail price only if valid
                               price = detail_price
                               self.logger.debug(f"상세 페이지 가격 확인: {price}")
                        except ValueError:
                            self.logger.warning(f"상세 페이지 가격 변환 오류: {price_text}")
            
            # 4. 수량별 가격 추출 - koryoproductpage.html table.quantity_price__table
            quantity_prices = {}
            qty_table = page.locator("table.quantity_price__table").first
            if qty_table.is_visible(timeout=5000): # Increased timeout
                try:
                    # 테이블 HTML 구조 확인용 로깅 (간략히)
                    # table_html = qty_table.inner_html(timeout=2000)
                    # self.logger.debug(f"수량별 가격 테이블 HTML(일부): {table_html[:100]}...")
                    
                    rows = qty_table.locator("tr").all()
                    if len(rows) >= 2:
                        qty_cells = rows[0].locator("td").all()
                        price_cells = rows[1].locator("td").all()
                        first_row_text = rows[0].text_content().strip()

                        if len(qty_cells) > 0 and ("수량" in first_row_text or "개" in first_row_text or qty_cells[0].text_content().strip().isdigit()):
                            self.logger.debug("수량별 가격 테이블 구조 확인: 행 분리형")
                            for i in range(min(len(qty_cells), len(price_cells))):
                                qty_text = qty_cells[i].text_content().strip()
                                price_text = price_cells[i].text_content().strip()
                                qty_match = re.search(r'(\d[\d,]*)', qty_text)
                                price_match = re.search(r'[\d,]+', price_text)
                                if qty_match and price_match:
                                    try:
                                        qty = int(qty_match.group(1).replace(",", ""))
                                        unit_price = float(price_match.group().replace(",", ""))
                                        if qty > 0 and unit_price > 0:
                                            quantity_prices[str(qty)] = unit_price
                                    except ValueError:
                                        self.logger.debug(f"수량/가격 변환 오류: Q='{qty_text}', P='{price_text}'")
                        else:
                             self.logger.debug("수량별 가격 테이블 구조 확인: 행 통합형 또는 기타")
                             # 다른 구조 처리 로직 (필요시 추가)

                    if quantity_prices:
                        self.logger.info(f"추출된 수량별 가격: {len(quantity_prices)}개 항목")
                    else:
                        self.logger.warning("수량별 가격을 추출하지 못했거나 테이블 구조 불일치")
                except Exception as qty_error:
                    self.logger.error(f"수량별 가격 추출 중 오류 발생: {qty_error}", exc_info=True)

            # 5. 상품 사양 추출 - koryoproductpage.html table.tbl_info
            specifications = {}
            specs_table = page.locator("table.tbl_info").first
            if specs_table.is_visible(timeout=5000): # Increased timeout
                 spec_rows = specs_table.locator("tr").all()
                 for row in spec_rows:
                    th = row.locator("th").first
                    td = row.locator("td").first
                    if th.count() > 0 and td.count() > 0:
                         key = th.text_content().strip().rstrip(':').strip()
                         value = td.text_content().strip()
                         if key and value:
                             specifications[key] = value
                 self.logger.debug(f"추출된 상품 사양: {len(specifications)}개 항목")

            # 6. 이미지 추출 - koryoproductpage.html #main_img, .thumnails img, .prd_detail img
            # 메인 이미지
            main_image_url = item.get("image_url") # Use list image as fallback
            main_img_elem = page.locator("#main_img").first
            if main_img_elem.is_visible(timeout=5000): # Increased timeout
                src = main_img_elem.get_attribute("src")
                if src:
                    # Check if it's a placeholder or valid image before overriding list image
                    if 'no_img' not in src.lower() and 'blank' not in src.lower():
                       # No need for urljoin here if we already did in the calling function
                       # if not src.startswith(("http://", "https://")):
                       #     src = urljoin(product_link, src)
                       main_image_url = src 
                       self.logger.debug(f"상세 페이지 메인 이미지 확인: {main_image_url}")

            # 이미지 갤러리 (썸네일 + 상세설명 이미지)
            image_gallery_list = []
            # 썸네일 (.product_picture .thumnails img)
            thumb_elements = page.locator(".product_picture .thumnails img").all()
            # 상세 설명 이미지 (.prd_detail img)
            detail_img_elements = page.locator(".prd_detail img").all()

            all_img_elements = thumb_elements + detail_img_elements

            for img_elem in all_img_elements:
                try: # Add try-except for individual image processing
                   if img_elem.is_visible(timeout=1000): # Short timeout for gallery images
                        src = img_elem.get_attribute("src")
                        if src and not self._is_ui_image(src):
                            # URL 정규화는 _validate_and_normalize_product 에서 처리
                            # if not src.startswith(("http://", "https://")):
                            #     src = urljoin(product_link, src)
                            if src not in image_gallery_list:
                                image_gallery_list.append(src)
                except Exception as img_err:
                    self.logger.debug(f"갤러리 이미지 처리 중 오류: {img_err}")


            self.logger.debug(f"추출된 이미지 갤러리: {len(image_gallery_list)}개 항목")

            # 7. 상세 설명 추출 - koryoproductpage.html div.prd_detail
            description = None
            desc_elem = page.locator("div.prd_detail").first
            if desc_elem.is_visible(timeout=5000): # Increased timeout
                try:
                    # HTML 형식으로 가져오기 (더 풍부한 정보)
                    desc_html = desc_elem.inner_html(timeout=5000) # Increased timeout
                    if desc_html and len(desc_html) > 50:
                        description = desc_html
                        self.logger.debug("상세 설명 HTML 추출 완료")
                except Exception as desc_error:
                    self.logger.warning(f"상세 설명 HTML 추출 오류: {desc_error}")
                    # 대체 텍스트 추출 시도 (필요시)
                    # desc_text = desc_elem.text_content(timeout=2000) ...
            
            # 7-1. 재고 상태 추출 (추가)
            stock_status = "In Stock"  # 기본값
            try:
                # 재고 관련 텍스트 검색
                stock_text_locators = [
                    page.locator('//span[contains(text(), "품절") or contains(text(), "재고") or contains(text(), "일시품절")]').first,
                    page.locator('//div[contains(text(), "품절") or contains(text(), "재고") or contains(text(), "일시품절")]').first,
                    page.locator('//td[contains(text(), "품절") or contains(text(), "재고") or contains(text(), "일시품절")]').first
                ]
                
                for locator in stock_text_locators:
                    if locator.count() > 0 and locator.is_visible(timeout=1000):
                        text = locator.text_content().strip()
                        if "품절" in text or "sold out" in text.lower():
                            stock_status = "Out of Stock"
                            self.logger.debug(f"품절 상태 감지: '{text}'")
                            break
                        elif "일시품절" in text or "temporarily out" in text.lower():
                            stock_status = "Temporarily Out of Stock"
                            self.logger.debug(f"일시품절 상태 감지: '{text}'")
                            break
                        elif "재고" in text and ("부족" in text or "없" in text):
                            stock_status = "Low Stock"
                            self.logger.debug(f"재고 부족 상태 감지: '{text}'")
                            break
                
                # 품절 버튼 또는 클래스 검색
                sold_out_elements = page.locator('.soldout, .out-of-stock, .sold-out, .item_soldout').all()
                if len(sold_out_elements) > 0:
                    for elem in sold_out_elements:
                        if elem.is_visible(timeout=1000):
                            stock_status = "Out of Stock"
                            self.logger.debug("품절 요소 감지")
                            break
            except Exception as stock_error:
                self.logger.warning(f"재고 상태 확인 중 오류: {stock_error}")
            
            # 8. 상태 확인 (가격, 이미지 기반)
            status = "OK"
            if price <= 0:
                status = "Price Not Found"
            elif not main_image_url or 'no_img' in main_image_url.lower():
                 status = "Image Not Found"

            # 9. 최종 제품 객체 생성 (검증/정규화는 별도 호출)
            try:
                product = Product(
                    id=product_id,
                    name=title,
                    price=price,
                    source="koryo",
                    url=product_link,
                    image_url=main_image_url, # Will be normalized later
                    product_code=product_code,
                    status=status,
                    stock_status=stock_status,  # 재고 상태 추가
                    # 기본 필드만 생성자에서 설정
                    quantity_prices=quantity_prices if quantity_prices else None
                )
                
                # 복잡한 속성은 객체 생성 후 별도로 설정
                # 사양 정보 설정
                if specifications and len(specifications) > 0:
                    product.specifications = specifications
                    
                # 설명 정보 설정
                if description:
                    product.description = description
                    
                # 이미지 갤러리 설정
                if image_gallery_list and len(image_gallery_list) > 0:
                    product.image_gallery = image_gallery_list
                    
                # 원본 데이터 설정
                if item:
                    product.original_input_data = item
                
                self.logger.info(f"상품 상세 정보 파싱 성공: {title}")
                return product # Return raw product, validation happens outside
            except Exception as product_creation_error:
                self.logger.error(f"상품 객체 생성 중 오류 발생: {product_creation_error}", exc_info=True)
                # 기본 최소 정보만으로 Product 객체 생성 시도
                try:
                    fallback_product = Product(
                        id=product_id,
                        name=title or "Extraction Error",
                        source="koryo",
                        price=price or 0.0,
                        url=product_link,
                        image_url=main_image_url,
                        status="Extraction Error"
                    )
                    self.logger.warning(f"대체 상품 객체 생성 성공: {fallback_product.name}")
                    return fallback_product
                except Exception as fallback_error:
                    self.logger.error(f"대체 상품 객체 생성도 실패: {fallback_error}", exc_info=True)
                    return None

        except Exception as e:
            self.logger.error(f"상품 상세 정보 추출 중 예외 발생: {product_link} - {e}", exc_info=True)
            # Return a minimal Product object on failure
            return Product(
                id=product_id,
                name=item.get("title", "Extraction Failed"),
                source="koryo",
                price=item.get("price", 0),
                url=product_link,
                image_url=item.get("image_url"),
                status="Extraction Failed"
            )

    def search_and_extract_lowest_price_product(self, query: str) -> Optional[Product]:
        """
        특정 검색어로 검색 후 가격 낮은 순으로 정렬하여 첫 번째 상품의 상세 정보를 추출하는 전체 흐름
        (Refined based on HTML analysis and logs)
        
        1. 메인 페이지에서 검색
        2. 검색 결과 페이지에서 가격 낮은순 정렬
        3. 첫 번째 상품 선택
        4. 상품 상세 페이지 이동 및 정보 추출
        
        Args:
            query: 검색어
            
        Returns:
            추출한 상품 정보 (Product 객체) 또는 None (실패 시)
        """
        self.logger.info(f"전체 흐름 시작: '{query}' 검색 및 최저가 상품 추출")
        
        # Default timeout for waits
        wait_timeout = 15000 
        
        with self.thread_context() as (browser, page):
            if not browser or not page:
                self.logger.error("브라우저 컨텍스트 생성 실패")
                return None
            
            try:
                # 1. 메인 페이지 접속
                self.logger.info(f"메인 페이지 접속: {self.base_url}")
                response = page.goto(self.base_url, wait_until='domcontentloaded', timeout=self.config.navigation_timeout)
                
                if not response or not response.ok:
                    self.logger.error(f"메인 페이지 접속 실패: {self.base_url}")
                    return None
                
                # 페이지 로딩 대기 (네트워크 활동 종료 대기)
                page.wait_for_load_state('networkidle', timeout=wait_timeout)
                
                # 2. 검색 수행 (안정성 강화)
                search_succeeded = False
                try:
                    # 검색창 찾기 (메인페이지 검색 - koryomainpage.html)
                    search_input = page.locator("#main_keyword").first
                    if search_input.is_visible(timeout=10000): # Increased timeout
                        # 검색어 입력
                        search_input.fill(query)
                        self.logger.info(f"검색어 입력: '{query}'")
                        
                        # 검색 버튼 클릭 시도 (koryomainpage.html)
                        search_button = page.locator(".search_btn_div img").first
                        if search_button.is_visible(timeout=5000):
                            search_button.click()
                            self.logger.info("검색 버튼 클릭")
                            # 페이지 이동 대기 (domcontentloaded가 더 빠름)
                            page.wait_for_load_state('domcontentloaded', timeout=wait_timeout) 
                            search_succeeded = True
                        else:
                            # 검색 버튼 없으면 Enter 키 시도
                            self.logger.info("검색 버튼 없음, 엔터키로 검색 제출 시도")
                            search_input.press("Enter", timeout=5000) # Added timeout
                            page.wait_for_load_state('domcontentloaded', timeout=wait_timeout)
                            search_succeeded = True
                            
                    else:
                        self.logger.warning("메인 검색창을 찾을 수 없음.")

                except Exception as search_error:
                    # 검색 오류 시 즉시 종료 (ERR_ABORTED 방지)
                    self.logger.error(f"메인 페이지 검색 중 오류 발생: {search_error}", exc_info=True)
                    return None # Do not proceed with direct URL navigation on error

                if not search_succeeded:
                     self.logger.error("메인 페이지 검색 실행 실패.")
                     return None

                # 스크린샷 저장 (디버깅용)
                try:
                    os.makedirs("screenshots", exist_ok=True)
                    safe_query = re.sub(r'[\/*?:"<>|]', "", query) # Sanitize query for filename
                    screenshot_path = f"screenshots/search_results_{safe_query}_{int(time.time())}.png"
                    page.screenshot(path=screenshot_path)
                    self.logger.info(f"검색 결과 스크린샷 저장: {screenshot_path}")
                except Exception as e:
                    self.logger.debug(f"스크린샷 저장 실패: {e}")
                
                # 3. 가격 낮은 순으로 정렬 (koryoaftersearch.html 기반)
                try:
                    self.logger.info("가격 낮은 순 정렬 시도")
                    
                    # 디버깅: 페이지 HTML 구조 로깅
                    try:
                        # 페이지 안정화를 위해 잠시 대기
                        page.wait_for_timeout(3000)
                        page.wait_for_load_state('domcontentloaded', timeout=10000)
                        
                        page_content = page.content()
                        self.logger.debug(f"페이지 HTML 내용 길이: {len(page_content)} 바이트")
                    except Exception as content_error:
                        self.logger.warning(f"페이지 HTML 내용 가져오기 실패: {content_error}")
                    
                    # 정렬 옵션 확인
                    order_items = page.locator('.order-item').all()
                    self.logger.debug(f"정렬 옵션 수: {len(order_items)}")
                    for i, item in enumerate(order_items):
                        try:
                            data_type = item.get_attribute('data-type')
                            text = item.text_content()
                            self.logger.debug(f"정렬 옵션 {i+1}: data-type='{data_type}', 텍스트='{text}'")
                        except Exception as e:
                            self.logger.debug(f"정렬 옵션 {i+1} 정보 추출 오류: {e}")
                    
                    # 정렬 옵션 (data-type="l") - koryoaftersearch.html
                    try:
                        # 먼저 페이지 안정화를 위해 잠시 대기
                        page.wait_for_timeout(2000)
                        
                        # 여러 방법으로 정렬 버튼 찾기 시도
                        # 방법 1: 직접 셀렉터로 찾기
                        sort_low_price = page.locator('.order-item[data-type="l"]').first
                        
                        # 방법 2: XPath로 텍스트 기반 검색
                        sort_by_text = page.locator('//div[contains(text(), "낮은가격") or contains(text(), "낮은 가격")]').first
                        
                        # 방법 3: 정렬 영역 내 클릭 시도
                        sort_container = page.locator('.order-wrap').first
                        
                        # 정렬 시도 및 로깅
                        self.logger.debug("정렬 버튼 찾기 및 클릭 시도")
                        
                        # 정렬 버튼이 나타날 때까지 최대 10초 대기
                        try:
                            page.wait_for_selector('.order-item[data-type="l"]', timeout=10000)
                            self.logger.debug("정렬 버튼(data-type='l') 발견")
                        except Exception as wait_error:
                            self.logger.warning(f"정렬 버튼 대기 중 오류: {wait_error}")
                            
                            # 다른 셀렉터로도 시도
                            try:
                                page.wait_for_selector('.order-wrap', timeout=5000)
                                self.logger.debug("정렬 컨테이너(.order-wrap) 발견")
                            except Exception as e:
                                self.logger.warning(f"정렬 컨테이너 대기 중 오류: {e}")
                        
                        # 정렬 버튼 클릭 시도 (여러 방법)
                        clicked = False
                        
                        # 방법 1 시도
                        if not clicked and sort_low_price.count() > 0:
                            try:
                                # 타임아웃 대폭 증가 및 강제 (force) 옵션 사용
                                sort_low_price.click(timeout=30000, force=True)
                                self.logger.info("가격 낮은 순 정렬 클릭 성공 (data-type 속성)")
                                clicked = True
                            except Exception as e:
                                self.logger.warning(f"방법 1 정렬 클릭 실패: {e}")
                        
                        # 방법 2 시도
                        if not clicked and sort_by_text.count() > 0:
                            try:
                                sort_by_text.click(timeout=30000, force=True)
                                self.logger.info("가격 낮은 순 정렬 클릭 성공 (텍스트 기반)")
                                clicked = True
                            except Exception as e:
                                self.logger.warning(f"방법 2 정렬 클릭 실패: {e}")
                        
                        # 방법 3 시도
                        if not clicked and sort_container.count() > 0:
                            try:
                                # 정렬 영역 좌측 클릭 (일반적으로 낮은가격순이 왼쪽에 위치)
                                sort_container.evaluate('el => { const rect = el.getBoundingClientRect(); const x = rect.left + 50; const y = rect.top + rect.height/2; document.elementFromPoint(x, y).click(); }')
                                self.logger.info("가격 낮은 순 정렬 클릭 성공 (컨테이너 내 위치 기반)")
                                clicked = True
                            except Exception as e:
                                self.logger.warning(f"방법 3 정렬 클릭 실패: {e}")
                        
                        # JavaScript로 직접 정렬 실행 (마지막 방법)
                        if not clicked:
                            try:
                                page.evaluate('''() => { 
                                    // 가격순 정렬 관련 JavaScript 함수 찾기 및 실행
                                    if (typeof sort_product === 'function') {
                                        sort_product('l'); 
                                        return true;
                                    } else if (document.querySelector('.order-item[data-type="l"]')) {
                                        document.querySelector('.order-item[data-type="l"]').click();
                                        return true;
                                    }
                                    return false;
                                }''')
                                self.logger.info("가격 낮은 순 정렬 JavaScript 실행 시도")
                                clicked = True
                            except Exception as e:
                                self.logger.warning(f"JavaScript 정렬 실행 실패: {e}")
                    except Exception as inner_sort_error:
                        self.logger.error(f"정렬 버튼 클릭 시도 중 오류: {inner_sort_error}")
                    
                    # 정렬 후 페이지가 완전히 로드될 때까지 대기
                    page.wait_for_timeout(3000)  # 일단 3초 강제 대기
                    try:
                        page.wait_for_load_state('networkidle', timeout=30000)
                        self.logger.debug("페이지 네트워크 활동 안정화 완료")
                    except Exception as e:
                        self.logger.warning(f"네트워크 안정화 대기 중 오류: {e}")
                        
                    try:
                        page.wait_for_load_state('domcontentloaded', timeout=30000)
                        self.logger.debug("페이지 DOM 로드 완료")
                    except Exception as e:
                        self.logger.warning(f"DOM 로딩 대기 중 오류: {e}")
                    
                    # 정렬 결과 확인용 스크린샷
                    try:
                        screenshot_path = f"screenshots/sorted_results_{safe_query}_{int(time.time())}.png"
                        page.screenshot(path=screenshot_path)
                        self.logger.info(f"정렬 결과 스크린샷 저장: {screenshot_path}")
                    except Exception as e:
                        self.logger.debug(f"정렬 스크린샷 저장 실패: {e}")
                    
                except Exception as sort_error:
                    self.logger.error(f"가격 정렬 중 오류 발생: {sort_error}", exc_info=True)
                    # 정렬 실패해도 일단 진행해볼 수 있음
                
                # 4. 첫 번째 상품 찾기 (koryoaftersearch.html 기반)
                self.logger.info("첫 번째 상품 찾기")
                
                # 상품 리스트 컨테이너 및 첫 번째 상품 - koryoaftersearch.html
                # 페이지 안정화를 위해 잠시 대기
                page.wait_for_timeout(2000)
                
                # 디버깅: 상품 목록 관련 정보 로깅
                try:
                    product_containers = page.locator('.product_lists').all()
                    self.logger.debug(f"상품 목록 컨테이너 수: {len(product_containers)}")
                    
                    # 여러 가능한 상품 요소 셀렉터 시도
                    all_selectors = [
                        '.product_lists .product', 
                        '.prd-list .prd-item',
                        'ul.prd_list li', 
                        '.best100_tab .product'
                    ]
                    
                    # 각 셀렉터별 결과 확인
                    for selector in all_selectors:
                        try:
                            items = page.locator(selector).all()
                            self.logger.debug(f"셀렉터 '{selector}'로 찾은 상품 수: {len(items)}")
                        except Exception as e:
                            self.logger.debug(f"셀렉터 '{selector}' 확인 중 오류: {e}")
                except Exception as debug_error:
                    self.logger.warning(f"상품 목록 디버깅 중 오류: {debug_error}")
                
                # 상품 목록이 나타날 때까지 대기 (타임아웃 증가)
                try:
                    page.wait_for_selector('.product_lists .product', timeout=20000)
                    self.logger.debug("상품 목록 셀렉터 발견됨")
                except Exception as wait_error:
                    self.logger.warning(f"상품 목록 대기 중 오류: {wait_error}")
                    # 다른 셀렉터도 시도
                    try:
                        for alt_selector in ['.prd-list .prd-item', 'ul.prd_list li']:
                            try:
                                page.wait_for_selector(alt_selector, timeout=5000)
                                self.logger.debug(f"대체 상품 목록 셀렉터 '{alt_selector}' 발견됨")
                                break
                            except Exception:
                                pass
                    except Exception as e:
                        self.logger.warning(f"대체 셀렉터 대기 중 오류: {e}")
                
                first_product_item = page.locator(".product_lists .product").first
                
                # 페이지에 상품이 있는지 확인 (타임아웃 증가 및 에러 처리 강화)
                try:
                    if not first_product_item.is_visible(timeout=20000):
                        self.logger.error("상품 목록(.product_lists .product)을 찾을 수 없음")
                        
                        # 대체 셀렉터 시도
                        alternative_selectors = [
                            '.prd-list .prd-item',
                            'ul.prd_list li', 
                            '.best100_tab .product'
                        ]
                        
                        for alt_selector in alternative_selectors:
                            alt_item = page.locator(alt_selector).first
                            if alt_item.count() > 0 and alt_item.is_visible(timeout=5000):
                                self.logger.info(f"대체 셀렉터 '{alt_selector}'로 첫 번째 상품 찾음")
                                first_product_item = alt_item
                                break
                        
                        # 여전히 상품 찾지 못했으면 종료
                        if not first_product_item.is_visible(timeout=5000):
                            return None
                except Exception as visibility_error:
                    self.logger.error(f"상품 가시성 확인 중 오류: {visibility_error}")
                    # JavaScript로 페이지에 상품이 있는지 확인
                    has_products = page.evaluate('''() => {
                        const selectors = ['.product_lists .product', '.prd-list .prd-item', 'ul.prd_list li', '.best100_tab .product'];
                        for (const selector of selectors) {
                            const items = document.querySelectorAll(selector);
                            if (items.length > 0) return true;
                        }
                        return false;
                    }''')
                    
                    if not has_products:
                        self.logger.error("JavaScript 검사로도 상품을 찾을 수 없음")
                        return None
                
                # 첫 번째 상품의 링크와 기본 정보 추출
                first_product_link = None
                
                # 상품 링크 시도 1: 이미지 링크 (.pic a) - koryoaftersearch.html
                img_link_elem = first_product_item.locator(".pic a").first
                if img_link_elem.is_visible(timeout=5000):
                    href = img_link_elem.get_attribute("href")
                    if href:
                        first_product_link = href
                        self.logger.info(f"첫 번째 상품 이미지 링크 찾음: {first_product_link}")

                # 상품 링크 시도 2: 상품명 링크 (.name a) - koryoaftersearch.html
                if not first_product_link:
                    name_link_elem = first_product_item.locator(".name a").first
                    if name_link_elem.is_visible(timeout=5000):
                         href = name_link_elem.get_attribute("href")
                         if href:
                             first_product_link = href
                             self.logger.info(f"첫 번째 상품 이름 링크 찾음: {first_product_link}")
                
                if not first_product_link:
                    self.logger.error("첫 번째 상품의 링크(.pic a 또는 .name a)를 찾을 수 없음")
                    return None
                
                # 상대 경로를 절대 경로로 변환
                if not first_product_link.startswith(("http://", "https://")):
                    first_product_link = urljoin(self.base_url, first_product_link)
                
                # 기본 정보 추출 (이름, 가격, 이미지) - koryoaftersearch.html
                name = ""
                price = 0.0
                image_url = ""
                
                # 상품명 (.name a)
                name_elem = first_product_item.locator(".name a").first 
                if name_elem.is_visible(timeout=3000):
                    name_text = name_elem.text_content()
                    if name_text:
                      name = name_text.strip()
                      
                # 가격 (.price)
                price_elem = first_product_item.locator(".price").first
                if price_elem.is_visible(timeout=3000):
                    price_text = price_elem.text_content()
                    if price_text:
                        price_match = re.search(r'[\d,]+', price_text)
                        if price_match:
                            try:
                                price = float(price_match.group().replace(",", ""))
                            except ValueError:
                                self.logger.warning(f"리스트 가격 변환 오류: {price_text}")
                
                # 이미지 (.pic img)
                img_elem = first_product_item.locator(".pic img").first
                if img_elem.is_visible(timeout=3000):
                    img_src = img_elem.get_attribute("src")
                    if img_src:
                        if not img_src.startswith(("http://", "https://")):
                            img_src = urljoin(self.base_url, img_src)
                        image_url = img_src
                
                # 상품 ID 추출 (URL에서)
                product_id = None
                id_match = re.search(r'no=(\d+)', first_product_link)
                if id_match:
                    product_id = id_match.group(1)
                else:
                    product_id = hashlib.md5(first_product_link.encode()).hexdigest()
                
                self.logger.info(f"첫 번째 상품 기본 정보 추출 완료: ID={product_id}, Name='{name}', Price={price}, Image='{image_url}'")

                # 5. 첫 번째 상품 상세 페이지로 이동
                self.logger.info(f"첫 번째 상품 상세 페이지로 이동: {first_product_link}")
                
                try:
                    page.goto(first_product_link, wait_until='domcontentloaded', timeout=self.config.navigation_timeout)
                    page.wait_for_load_state('networkidle', timeout=wait_timeout)
                    
                    # 상세 페이지 스크린샷 저장
                    try:
                        screenshot_path = f"screenshots/first_product_detail_{product_id}_{int(time.time())}.png"
                        page.screenshot(path=screenshot_path)
                        self.logger.info(f"첫 번째 상품 상세 스크린샷 저장: {screenshot_path}")
                    except Exception as e:
                        self.logger.debug(f"상세 스크린샷 저장 실패: {e}")
                    
                    # 6. 상품 상세 정보 추출 (koryoproductpage.html 기반)
                    item_data = {
                        "title": name, # Pass basic info extracted from list
                        "price": price,
                        "link": first_product_link,
                        "image_url": image_url,
                        "product_id": product_id
                    }
                    
                    detailed_product = self._extract_product_details(page, item_data)
                    
                    if detailed_product:
                        # Validate and normalize (especially image URLs) before returning
                        if self._validate_and_normalize_product(detailed_product):
                            self.logger.info(f"최저가 상품 상세 정보 추출 및 검증 성공: {detailed_product.name}")
                            return detailed_product
                        else:
                           self.logger.error("상품 상세 정보 검증 실패")
                           # Return the raw product anyway? Or None? Let's return raw for now.
                           return detailed_product 
                    else:
                        self.logger.error("상품 상세 정보 추출 실패")
                        return None
                        
                except Exception as detail_error:
                    self.logger.error(f"상품 상세 페이지 접근 또는 정보 추출 중 오류 발생: {detail_error}", exc_info=True)
                    return None
                
            except Exception as e:
                self.logger.error(f"최저가 상품 검색 및 추출 흐름 중 오류 발생: {e}", exc_info=True)
                return None

    def _is_ui_image(self, url: str) -> bool:
        """URL이 UI 요소 이미지인지 간단히 확인"""
        url_lower = url.lower()
        ui_patterns = [
            'btn_', 'icon_', 'dot_', 'bottom_', '/bbs/image/',
            'guide1.jpg', 's1.png', '/common/', '/layout/', '/banner/',
            '.gif'
        ]
        return any(pattern in url_lower for pattern in ui_patterns)

    # --- Playwright 기반 스크래핑 메서드 --- 

    def get_product(self, product_id: str) -> Optional[Product]:
        """상품 ID로 상품 정보 가져오기"""
        self.logger.info(f"상품 정보 조회 시작: {product_id}")
        
        # 실제 상품 상세 URL 패턴 사용
        product_url = f"{self.base_url}/ez/mall.php?query=view&no={product_id}"
        
        with self.thread_context() as (browser, page):
            if not browser or not page:
                self.logger.error(f"브라우저 컨텍스트 생성 실패: {product_id}")
                return None
                
            try:
                self.logger.info(f"상품 페이지 접속: {product_url}")
                response = page.goto(
                    product_url, 
                    wait_until='domcontentloaded',
                    timeout=self.config.navigation_timeout
                )
                
                if not response or not response.ok:
                    self.logger.error(f"상품 페이지 접속 실패: {product_url}")
                    return None
                
                # 상품 정보 추출
                item_data = {
                    "link": product_url,
                    "product_id": product_id
                }
                
                product = self._extract_product_details(page, item_data)
                if product:
                    self.logger.info(f"상품 정보 추출 성공: {product.name}")
                    return product
            except Exception as e:
                self.logger.error(f"상품 정보 추출 중 오류 발생: {e}", exc_info=True)
                return None
                    