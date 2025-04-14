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

from playwright.sync_api import sync_playwright, Playwright, Browser, Page, Locator, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

# Add imports for caching
from utils.caching import FileCache, cache_result

from ..data_models import Product, ProductStatus
from .base_multi_layer_scraper import BaseMultiLayerScraper
from .extraction_strategy import ExtractionStrategy, strategies
from .selectors import KORYO_SELECTORS

# Add imports for threading
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ScraperConfig:
    """스크래퍼 설정을 위한 데이터 클래스"""
    max_retries: int = 3 # 네트워크 오류 재시도 횟수
    timeout: int = 60000 # Playwright 기본 타임아웃 (ms)
    navigation_timeout: int = 60000 # 페이지 이동 타임아웃 (ms)
    wait_timeout: int = 30000 # 특정 요소 대기 타임아웃 (ms)
    request_delay: float = 1.0 # 작업 사이 지연 시간 (초)
    headless: bool = True # Headless 모드 사용 여부
    debug: bool = False # 디버그 로깅 활성화
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"


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
        debug: bool = False
    ):
        # 설정 초기화
        self.config = config or ScraperConfig()
        
        # debug 파라미터 처리
        if debug:
            self.config.debug = debug
        
        # 타임아웃 설정 적용
        if read_timeout is not None:
            self.config.timeout = read_timeout
            
        if connect_timeout is not None:
            self.config.navigation_timeout = connect_timeout
            
        # 캐시 TTL 설정
        self.cache_ttl = cache_ttl or 3600  # 기본값 1시간
            
        # BaseScraper 초기화
        super().__init__(max_retries=max_retries or self.config.max_retries, 
                        timeout=timeout or self.config.timeout, 
                        cache=cache)
        
        self.logger = logging.getLogger(__name__)
        # 실제 작동하는 도메인으로 수정
        self.base_url = "https://adpanchok.co.kr"
        self.mall_url = f"{self.base_url}/ez/mall.php"
        self.search_url = f"{self.base_url}/ez/goods/goods_search.php"
        
        # Playwright 인스턴스 및 리소스 초기화
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        
        # Thread-local storage 초기화
        self._thread_local = threading.local()
        self._thread_local.playwright = None
        self._thread_local.browser = None
        self._thread_local.context = None
        self._thread_local.page = None

        # 실제 카테고리 예시 추가
        self.default_categories = [
            ("볼펜/사무용품", "013001001"),  # 실제 카테고리 코드
            ("기념타월/수건", "004006001"),  # 실제 카테고리 코드
            ("전자/디지털", "013004000")     # 실제 카테고리 코드
        ]

        # 검색 URL 패턴 수정
        self.base_search_urls = [
            f"{self.base_url}/ez/mall.php"  # 실제 작동하는 URL 패턴
        ]
        
        # Selectors for product extraction
        self.title_selectors = [".name a", ".name", "h3.name", ".goods_name", ".product_name"]
        self.price_selectors = [".price", "div.price", "p.price", "strong.price"]
        self.image_selectors = [".pic img", ".img img", "div.pic img", ".thumb img"]
        
        # Patterns for extracting data
        self.patterns = {
            "price_number": re.compile(r'[\d,]+'),
            "product_id": re.compile(r'no=(\d+)')
        }
        
        # Selectors dictionary
        self.selectors = KORYO_SELECTORS

    def __del__(self):
        """리소스 정리"""
        try:
            self.close()
        except Exception as e:
            self.logger.warning(f"Error in __del__: {e}")

    def close(self):
        """모든 리소스 정리"""
        try:
            if hasattr(self, '_thread_local'):
                self._close_thread_context()
            
            if hasattr(self, '_page') and self._page:
                try:
                    self._page.close()
                except Exception as e:
                    self.logger.warning(f"Error closing page: {e}")
                self._page = None
            
            if hasattr(self, '_browser') and self._browser and self._browser.is_connected():
                try:
                    self._browser.close()
                except Exception as e:
                    self.logger.warning(f"Error closing browser: {e}")
                self._browser = None
            
            if hasattr(self, '_playwright') and self._playwright:
                try:
                    self._playwright.stop()
                except Exception as e:
                    self.logger.warning(f"Error stopping playwright: {e}")
                self._playwright = None
        except Exception as e:
            self.logger.error(f"Error in close(): {e}", exc_info=True)

    def _create_new_context(self) -> tuple[Optional[Browser], Optional[Page]]:
        """브라우저 컨텍스트 및 페이지 생성 (스레드 안전)"""
        try:
            # 설정 값
            viewport = {'width': 1920, 'height': 1080}
            user_agent = self.config.user_agent
            
            # 플레이라이트 초기화
            playwright = sync_playwright().start()
            
            # 브라우저 시작 (크로미움 기반)
            browser = playwright.chromium.launch(
                headless=self.config.headless,
                args=[
                    '--disable-web-security',
                    '--no-sandbox',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-dev-shm-usage',  # 메모리 문제 방지
                    '--disable-gpu',  # GPU 가속 비활성화
                    '--disable-setuid-sandbox',  # 추가 안정성
                    '--disable-extensions',  # 확장 프로그램 비활성화
                    '--disable-infobars'  # 정보 표시줄 비활성화
                ]
            )
            
            # 컨텍스트 생성 (쿠키/세션 컨테이너) - 프록시 제거
            context = browser.new_context(
                user_agent=user_agent,
                viewport=viewport,
                ignore_https_errors=True,  # SSL 오류 무시
                java_script_enabled=True
            )
            
            # 타임아웃 설정 (짧게 유지)
            context.set_default_timeout(self.config.timeout)
            context.set_default_navigation_timeout(self.config.navigation_timeout)
            
            # 쿠키 설정 간소화
            context.add_cookies([
                {
                    "name": "accept_cookies",
                    "value": "true",
                    "domain": "koreagift.com",
                    "path": "/"
                }
            ])
            
            # 새 페이지 생성
            page = context.new_page()
            
            # 스크롤 헬퍼 스크립트 추가
            page.add_init_script("""
                window.autoScroll = function(duration) {
                    return new Promise((resolve) => {
                        const scrollHeight = document.body.scrollHeight;
                        const step = scrollHeight / 20;
                        let current = 0;
                        
                        const scroller = setInterval(() => {
                            window.scrollBy(0, step);
                            current += step;
                            
                            if (current >= scrollHeight) {
                                clearInterval(scroller);
                                resolve();
                            }
                        }, duration / 20);
                    });
                };
            """)
            
            # 스레드 로컬 스토리지에 저장
            self._thread_local.playwright = playwright
            self._thread_local.browser = browser
            self._thread_local.context = context
            self._thread_local.page = page
            
            return browser, page
            
        except Exception as e:
            self.logger.error(f"브라우저 컨텍스트 생성 실패: {e}", exc_info=True)
            # 자원 정리
            self._close_thread_context()
            return None, None
            
    @contextmanager
    def get_browser_context(self) -> Generator[tuple[Optional[Browser], Optional[Page]], None, None]:
        """컨텍스트 매니저를 사용해 브라우저 자원을 안전하게 관리"""
        browser, page = self._create_new_context()
        try:
            yield browser, page
        finally:
            # 항상 자원을 정리하도록 보장
            if page and not page.is_closed():
                try:
                    page.close()
                except Exception as e:
                    self.logger.warning(f"페이지 종료 오류: {e}")
            
            self._close_thread_context()
    
    def search_products_batch(self, queries: List[str], max_items_per_query: int = 10) -> Dict[str, List[Product]]:
        """여러 검색어를 일괄 처리하여 검색 결과 반환 (멀티스레딩 활용)"""
        if not queries:
            return {}
            
        results = {}
        
        # 멀티스레딩 활용 (동시에 여러 검색 수행)
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            # 각 검색어에 대한 작업 제출
            future_to_query = {
                executor.submit(self.search_product, query, max_items_per_query): query
                for query in queries
            }
            
            # 결과 수집
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    query_results = future.result()
                    results[query] = query_results
                    self.logger.info(f"검색 일괄 처리: '{query}' 결과 {len(query_results)}개")
                except Exception as e:
                    self.logger.error(f"검색 일괄 처리 중 오류 ('{query}'): {e}")
                    results[query] = [self._create_no_match_product(query, is_error=True)]
        
        return results

    def _close_thread_context(self):
        """Close and clean up thread-local browser context"""
        try:
            if hasattr(self._thread_local, 'page') and self._thread_local.page:
                try:
                    self._thread_local.page.close()
                except Exception as e:
                    self.logger.warning(f"Error closing thread-local page: {e}")
                self._thread_local.page = None

            if hasattr(self._thread_local, 'context') and self._thread_local.context:
                try:
                    self._thread_local.context.close()
                except Exception as e:
                    self.logger.warning(f"Error closing thread-local context: {e}")
                self._thread_local.context = None

            if hasattr(self._thread_local, 'browser') and self._thread_local.browser and self._thread_local.browser.is_connected():
                try:
                    self._thread_local.browser.close()
                except Exception as e:
                    self.logger.warning(f"Error closing thread-local browser: {e}")
                self._thread_local.browser = None

            if hasattr(self._thread_local, 'playwright') and self._thread_local.playwright:
                try:
                    self._thread_local.playwright.stop()
                except Exception as e:
                    self.logger.warning(f"Error stopping thread-local playwright: {e}")
                self._thread_local.playwright = None
        except Exception as e:
            self.logger.error(f"Error cleaning up thread context: {e}", exc_info=True)

    @contextmanager
    def thread_context(self) -> Generator[tuple[Optional[Browser], Optional[Page]], None, None]:
        """Context manager for creating and cleaning up thread-local browser context"""
        browser, page = self._create_new_context()
        try:
            # 네트워크 타임아웃 값 증가 (실제 환경에 맞게 조정)
            if page:
                page.set_default_timeout(60000)
                page.set_default_navigation_timeout(60000)
            yield browser, page
        finally:
            if page and not page.is_closed():
                try:
                    page.close()
                except Exception as e:
                    self.logger.warning(f"Error closing page in thread_context: {e}")
            self._close_thread_context()

    def _get_page(self) -> Optional[Page]:
         """초기화된 Playwright Page 객체를 반환하거나 새로 초기화"""
         if not self._page or self._page.is_closed():
             if not self.init_playwright():
                 return None
         # Also check browser connection status
         if not self._browser or not self._browser.is_connected():
             self.logger.warning("Browser is disconnected. Attempting to re-initialize.")
             if not self.init_playwright():
                 self.logger.error("Failed to re-initialize Playwright.")
                 return None
         return self._page

    # --- Helper Methods --- 

    def _safe_get_text(self, locator: Locator, timeout: Optional[int] = None) -> Optional[str]:
        """타임아웃 및 오류 처리하여 Locator의 텍스트 콘텐츠 가져오기"""
        try:
            if not locator:
                self.logger.debug("Locator is None")
                return None
                
            if not locator.is_visible(timeout=1000):
                self.logger.debug("Locator is not visible")
                return None
                
            text = locator.text_content(timeout=timeout or self.config.wait_timeout)
            if not text:
                self.logger.debug("No text content found")
                return None
                
            return text.strip()
            
        except (PlaywrightTimeoutError, PlaywrightError) as e:
            self.logger.debug(f"Timeout or error getting text content: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting text content: {e}", exc_info=self.config.debug)
            return None

    def _safe_get_attribute(self, locator: Locator, attribute: str, timeout: Optional[int] = None) -> Optional[str]:
        """타임아웃 및 오류 처리하여 Locator의 속성 값 가져오기"""
        try:
            if not locator:
                self.logger.debug("Locator is None")
                return None
                
            if not locator.is_visible(timeout=1000):
                self.logger.debug("Locator is not visible")
                return None
                
            value = locator.get_attribute(attribute, timeout=timeout or self.config.wait_timeout)
            if not value:
                self.logger.debug(f"No attribute '{attribute}' found")
                return None
                
            return value.strip()
            
        except (PlaywrightTimeoutError, PlaywrightError) as e:
            self.logger.debug(f"Timeout or error getting attribute '{attribute}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting attribute '{attribute}': {e}", exc_info=self.config.debug)
            return None

    def _wait_for_load_state(self, page: Page, state: str = 'networkidle', timeout: Optional[int] = None):
        """페이지 로드 상태 대기 (오류 처리 포함)"""
        try:
            page.wait_for_load_state(state, timeout=timeout or self.config.navigation_timeout)
            self.logger.debug(f"Page reached load state: '{state}'")
        except (PlaywrightTimeoutError, PlaywrightError) as e:
             self.logger.warning(f"Timeout or error waiting for page load state '{state}': {e}")
             # 오류 발생해도 계속 진행할 수 있도록 예외 처리

    def _wait_for_selector(self, page: Page, selector: str, state: str = 'visible', timeout: Optional[int] = None) -> Optional[Locator]:
        """특정 셀렉터 대기 (오류 처리 포함)"""
        try:
            locator = page.locator(selector)
            locator.wait_for(state=state, timeout=timeout or self.config.wait_timeout)
            self.logger.debug(f"Selector '{selector}' is now {state}.")
            return locator
        except (PlaywrightTimeoutError, PlaywrightError) as e:
             self.logger.warning(f"Timeout or error waiting for selector '{selector}' to be {state}: {e}")
             return None

    def _click_locator(self, locator: Locator, force: bool = False, timeout: Optional[int] = None) -> bool:
         """안전하게 로케이터 클릭"""
         try:
             locator.click(force=force, timeout=timeout or self.config.wait_timeout)
             self.logger.debug("Clicked locator successfully.")
             return True
         except (PlaywrightTimeoutError, PlaywrightError) as e:
             self.logger.warning(f"Timeout or error clicking locator: {e}")
             return False

    # --- 데이터 추출 메서드 (Playwright Locator 기반) --- 

    def _extract_list_item(self, element_locator: Locator) -> Optional[Dict]:
        """제품 목록 항목 Locator에서 기본 정보 추출"""
        product_data = {}
        try:
            # 제품명 추출
            title = None
            for selector in self.title_selectors:
                title_locator = element_locator.locator(selector).first
                title = self._safe_get_text(title_locator)
                if title:
                    break
                    
            if not title:
                self.logger.debug("Could not extract title from list item")
                return None
                
            product_data["title"] = title

            # 링크 추출
            link = None
            for selector in self.selectors["product_link_list"]["selector"].split(","):
                selector = selector.strip()
                if not selector:
                    continue
                    
                link_locator = element_locator.locator(selector).first
                link = self._safe_get_attribute(link_locator, "href")
                if link:
                    break
                    
            if link:
                product_data["link"] = urljoin(self.base_url, link)
            else:
                self.logger.debug("Could not extract link from list item")

            # 가격 추출
            price = 0.0
            for selector in self.price_selectors:
                price_locator = element_locator.locator(selector).first
                price_text = self._safe_get_text(price_locator)
                if price_text:
                    price_match = self.patterns["price_number"].search(price_text)
                    if price_match:
                        try:
                            price = float(price_match.group().replace(",", ""))
                            break
                        except ValueError:
                            self.logger.debug(f"Could not parse price text: {price_text}")
                            
            product_data["price"] = price

            # 이미지 추출
            image = None
            for selector in self.image_selectors:
                img_locator = element_locator.locator(selector).first
                image = self._safe_get_attribute(img_locator, "src")
                if image:
                    break
                    
            if image:
                product_data["image"] = urljoin(self.base_url, image)
            else:
                self.logger.debug("Could not extract image from list item")

            return product_data
            
        except Exception as e:
            self.logger.error(f"Error extracting list item: {e}", exc_info=self.config.debug)
            return None

    def _extract_product_details(self, page: Page, item: Dict) -> Optional[Product]:
        """상세 페이지에서 제품 정보 추출 (개선 버전)"""
        product_id = item.get("product_id")
        product_link = item.get("link")
        
        if not product_link:
            self.logger.warning("상품 링크 정보 없음")
            return None
            
        if not product_id:
            product_id = hashlib.md5(product_link.encode()).hexdigest()
        
        # koryoproductpage.html 분석 기반 정확한 셀렉터 사용
        try:
            # 1. 상품명 추출
            title = item.get("title", "Unknown Product")  # 기본값
            
            # 상세 페이지 상품명 (보다 정확함)
            title_locator = page.locator(".product_name").first
            if title_locator.is_visible(timeout=2000):
                detail_title = title_locator.text_content(timeout=2000)
                if detail_title and len(detail_title.strip()) > 0:
                    title = detail_title.strip()
                    self.logger.debug(f"상세 페이지 상품명: {title}")
            
            # 2. 상품 코드 추출
            product_code = None
            
            # 상품 코드 텍스트 찾기 (일반적으로 "상품코드: XXXXX" 형식)
            code_locators = [
                page.locator('//div[contains(text(), "상품코드")]').first,
                page.locator('//span[contains(text(), "상품코드")]').first,
                page.locator('//td[contains(text(), "상품코드")]').first
            ]
            
            for locator in code_locators:
                if locator.count() > 0 and locator.is_visible(timeout=1000):
                    code_text = locator.text_content(timeout=1000)
                    if code_text:
                        # 정규식으로 상품 코드만 추출
                        code_match = re.search(r'상품코드\s*[:\-]?\s*([A-Za-z0-9_-]+)', code_text)
                        if code_match:
                            product_code = code_match.group(1).strip()
                            self.logger.debug(f"상품 코드: {product_code}")
                            break
            
            # 3. 가격 정보 추출
            price = item.get("price", 0.0)  # 기본값
            
            # 상세 페이지 가격 (#main_price가 가장 정확함)
            price_locator = page.locator("#main_price").first
            if price_locator.is_visible(timeout=2000):
                price_text = price_locator.text_content(timeout=2000)
                if price_text:
                    # 숫자만 추출
                    price_match = re.search(r'[\d,]+', price_text)
                    if price_match:
                        try:
                            price = float(price_match.group().replace(",", ""))
                            self.logger.debug(f"상품 가격: {price}")
                        except ValueError:
                            self.logger.warning(f"가격 변환 오류: {price_text}")
            
            # 4. 수량별 가격 추출
            quantity_prices = {}
            qty_table = page.locator("table.quantity_price__table").first
            
            if qty_table.is_visible(timeout=2000):
                try:
                    # 테이블 HTML 구조 확인용 로깅
                    table_html = qty_table.inner_html(timeout=2000)
                    self.logger.debug(f"수량별 가격 테이블: {table_html[:200]}...")
                    
                    # 테이블 행 가져오기
                    rows = qty_table.locator("tr").all()
                    
                    if len(rows) >= 2:  # 최소 2개 행 (헤더 + 가격)
                        # 방법 1: 첫 번째 행이 수량, 두 번째 행이 가격인 경우 (일반적인 경우)
                        qty_cells = rows[0].locator("td").all()
                        price_cells = rows[1].locator("td").all()
                        
                        # 수량 행에 "수량" 또는 "개" 등의 텍스트가 있는지 확인
                        first_row_text = rows[0].text_content().strip()
                        if len(qty_cells) > 0 and ("수량" in first_row_text or "개" in first_row_text):
                            self.logger.debug("테이블 구조: 첫 번째 행=수량, 두 번째 행=가격")
                            
                            # 각 열 처리
                            for i in range(min(len(qty_cells), len(price_cells))):
                                qty_text = qty_cells[i].text_content().strip()
                                price_text = price_cells[i].text_content().strip()
                                
                                if qty_text and price_text:
                                    # 수량에서 숫자만 추출 (콤마 제거)
                                    qty_match = re.search(r'(\d[\d,]*)', qty_text)
                                    if qty_match:
                                        try:
                                            qty = int(qty_match.group(1).replace(",", ""))
                                            
                                            # 가격에서 숫자만 추출
                                            price_match = re.search(r'[\d,]+', price_text)
                                            if price_match:
                                                try:
                                                    unit_price = float(price_match.group().replace(",", ""))
                                                    quantity_prices[str(qty)] = unit_price
                                                    self.logger.debug(f"수량 {qty}개: {unit_price}원 추출")
                                                except ValueError:
                                                    self.logger.warning(f"가격 변환 오류: '{price_text}'")
                                        except ValueError:
                                            self.logger.warning(f"수량 변환 오류: '{qty_text}'")
                        else:
                            # 방법 2: 각 행에 수량과 가격이 같이 있는 경우
                            self.logger.debug("테이블 구조: 각 행에 수량과 가격이 함께 있음")
                            
                            for row in rows:
                                cells = row.locator("td").all()
                                if len(cells) >= 2:  # 최소 두 개의 셀 (수량, 가격)
                                    # 첫 번째 셀이 수량, 두 번째 셀이 가격인지 확인
                                    qty_text = cells[0].text_content().strip()
                                    price_text = cells[1].text_content().strip()
                                    
                                    # 수량 추출 (숫자와 '개' 등의 단위가 함께 있을 수 있음)
                                    qty_match = re.search(r'(\d[\d,]+)(?:개|EA|ea|pcs)?', qty_text)
                                    if qty_match:
                                        try:
                                            qty = int(qty_match.group(1).replace(",", ""))
                                            
                                            # 가격 추출
                                            price_match = re.search(r'[\d,]+', price_text)
                                            if price_match:
                                                try:
                                                    unit_price = float(price_match.group().replace(",", ""))
                                                    quantity_prices[str(qty)] = unit_price
                                                    self.logger.debug(f"수량 {qty}개: {unit_price}원 추출 (행 기준)")
                                                except ValueError:
                                                    pass
                                        except ValueError:
                                            pass
                    
                    # 방법 3: 규칙적인 수량 패턴 확인 및 사용자 제공 데이터 활용
                    if not quantity_prices:
                        # 일반적인 수량 패턴
                        common_quantities = [5000, 3000, 1000, 500, 300, 200, 100, 50]
                        
                        # 테이블의 모든 텍스트 추출
                        table_text = qty_table.text_content()
                        
                        # 각 수량별로 검사
                        for qty in common_quantities:
                            # 수량 뒤에 나오는 가격 패턴 찾기
                            pattern = fr'{qty:,}개?\s*[\r\n]*([0-9,]+)'
                            match = re.search(pattern, table_text.replace(',', ''))
                            if match:
                                try:
                                    price_str = match.group(1).replace(',', '')
                                    price = float(price_str)
                                    quantity_prices[str(qty)] = price
                                    self.logger.debug(f"패턴 기반 추출 - 수량 {qty}개: {price}원")
                                except (ValueError, IndexError):
                                    pass
                    
                    # 최종 수량별 가격 확인
                    if quantity_prices:
                        self.logger.info(f"추출된 수량별 가격: {len(quantity_prices)}개 항목")
                        for qty, price in sorted(quantity_prices.items(), key=lambda x: int(x[0]), reverse=True):
                            self.logger.debug(f"  수량 {qty}개: {price:,}원")
                    else:
                        self.logger.warning("수량별 가격을 추출하지 못했습니다.")
                
                except Exception as qty_error:
                    self.logger.error(f"수량별 가격 추출 중 오류: {qty_error}", exc_info=True)
            
            # 5. 상품 사양 추출
            specifications = {}
            specs_table = page.locator("table.tbl_info").first
            
            if specs_table.is_visible(timeout=2000):
                # 테이블 행 가져오기
                spec_rows = specs_table.locator("tr").all()
                
                for row in spec_rows:
                    # <tr><th>항목</th><td>값</td></tr> 구조
                    th = row.locator("th").first
                    td = row.locator("td").first
                    
                    if th.count() > 0 and td.count() > 0:
                        key = th.text_content().strip()
                        value = td.text_content().strip()
                        
                        if key and value:
                            # 키에서 콜론(:) 제거
                            key = key.rstrip(':').strip()
                            specifications[key] = value
                
                self.logger.debug(f"상품 사양: {len(specifications)}개 항목")
            
            # 6. 이미지 추출
            # 메인 이미지
            main_image_url = item.get("image_url")  # 기본값
            
            main_img = page.locator("#main_img").first
            if main_img.is_visible(timeout=2000):
                src = main_img.get_attribute("src")
                if src:
                    if not src.startswith(("http://", "https://")):
                        src = urljoin(product_link, src)
                    main_image_url = src
            
            # 이미지 갤러리 (썸네일 및 추가 이미지)
            image_gallery_list = []
            thumbs = page.locator(".product_picture .thumnails img").all()
            
            # 썸네일 이미지
            for thumb in thumbs:
                if thumb.is_visible(timeout=500):
                    src = thumb.get_attribute("src")
                    if src and not self._is_ui_image(src):
                        if not src.startswith(("http://", "https://")):
                            src = urljoin(product_link, src)
                        if src not in image_gallery_list:
                            image_gallery_list.append(src)
            
            # 상세 설명 내 이미지
            detail_imgs = page.locator("div.prd_detail img").all()
            for img in detail_imgs:
                if img.is_visible(timeout=500):
                    src = img.get_attribute("src")
                    if src and not self._is_ui_image(src):
                        if not src.startswith(("http://", "https://")):
                            src = urljoin(product_link, src)
                        if src not in image_gallery_list:
                            image_gallery_list.append(src)
            
            # 7. 상세 설명 추출
            description = None
            desc_elem = page.locator("div.prd_detail").first
            
            if desc_elem.is_visible(timeout=2000):
                try:
                    # HTML 형식으로 가져오기 (서식 유지)
                    desc_html = desc_elem.inner_html(timeout=3000)
                    if desc_html and len(desc_html) > 50:  # 최소 길이 확인
                        description = desc_html
                except Exception as desc_error:
                    self.logger.warning(f"상세 설명 HTML 추출 오류: {desc_error}")
                    # 텍스트 형식으로 대체
                    desc_text = desc_elem.text_content(timeout=2000)
                    if desc_text and len(desc_text) > 50:
                        description = desc_text
            
            # 8. 상태 확인
            status = "OK"
            
            if not main_image_url:
                status = "Image Not Found"
            elif price <= 0:
                status = "Price Not Found"
            
            # 9. 최종 제품 객체 생성
            # image_gallery 파라미터 이슈 수정 - _image_gallery 속성에 직접 할당
            product = Product(
                id=product_id,
                name=title,
                price=price,
                source="koryo",
                url=product_link,
                image_url=main_image_url,
                product_code=product_code,
                status=status,
            )
            
            # _image_gallery에 직접 할당 (property를 통해)
            if image_gallery_list:
                product.image_gallery = image_gallery_list
                
            # 기타 속성도 필요에 따라 설정
            if quantity_prices:
                product.quantity_prices = quantity_prices
                
            if specifications:
                product.specifications = specifications
                
            if description:
                product.description = description
                
            # 원본 데이터 저장
            product.original_input_data = item
            
            self.logger.info(f"상품 상세 정보 추출 성공: {title}")
            return product
            
        except Exception as e:
            self.logger.error(f"상품 상세 정보 추출 오류: {product_link} - {e}", exc_info=True)
            
            # 최소한의 제품 객체 반환
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
        
        with self.thread_context() as (browser, page):
            if not browser or not page:
                self.logger.error("브라우저 컨텍스트 생성 실패")
                return None
            
            try:
                # 1. 메인 페이지 접속
                self.logger.info(f"메인 페이지 접속: {self.base_url}")
                response = page.goto(self.base_url, wait_until='domcontentloaded', timeout=30000)
                
                if not response or not response.ok:
                    self.logger.error(f"메인 페이지 접속 실패: {self.base_url}")
                    return None
                
                # 페이지 로딩 대기 (네트워크 활동 종료 대기)
                self._wait_for_load_state(page)
                
                # 2. 검색 수행
                try:
                    # 검색창 찾기 (메인페이지 검색)
                    search_input = page.locator("#main_keyword").first
                    if search_input.is_visible(timeout=5000):
                        # 검색어 입력
                        search_input.fill(query)
                        self.logger.info(f"검색어 입력: '{query}'")
                        
                        # 검색 버튼 클릭
                        search_button = page.locator(".search_btn_div img").first
                        if search_button.is_visible(timeout=3000):
                            search_button.click()
                            self.logger.info("검색 버튼 클릭")
                            # 페이지 로딩 대기
                            page.wait_for_load_state('networkidle', timeout=15000)
                        else:
                            # 엔터키 입력으로 검색
                            search_input.press("Enter")
                            self.logger.info("엔터키로 검색 제출")
                            page.wait_for_load_state('networkidle', timeout=15000)
                    else:
                        # 검색창을 찾을 수 없는 경우 직접 URL 이동
                        self.logger.warning("메인 검색창을 찾을 수 없어 직접 URL 이동")
                        encoded_query = urllib.parse.quote(query)
                        search_url = f"{self.base_url}/ez/mall.php?search_str={encoded_query}"
                        page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                        page.wait_for_load_state('networkidle', timeout=15000)
                except Exception as search_error:
                    self.logger.error(f"메인 페이지 검색 중 오류: {search_error}")
                    # 직접 URL 이동으로 대체
                    encoded_query = urllib.parse.quote(query)
                    search_url = f"{self.base_url}/ez/mall.php?search_str={encoded_query}"
                    page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                
                # 스크린샷 저장 (디버깅용)
                try:
                    os.makedirs("screenshots", exist_ok=True)
                    screenshot_path = f"screenshots/search_results_{query}_{int(time.time())}.png"
                    page.screenshot(path=screenshot_path)
                    self.logger.info(f"검색 결과 스크린샷 저장: {screenshot_path}")
                except Exception as e:
                    self.logger.debug(f"스크린샷 저장 실패: {e}")
                
                # 3. 가격 낮은 순으로 정렬
                try:
                    # koryoaftersearch.html 분석 기반 정렬 옵션 클릭
                    # 정렬 옵션은 일반적으로 data-type="l" 속성을 가진 요소임 (낮은가격순)
                    self.logger.info("가격 낮은 순 정렬 시도")
                    
                    # 정렬 옵션 시도 1: data-type 속성 이용
                    sort_low_price = page.locator('.order-item[data-type="l"]').first
                    
                    if sort_low_price.is_visible(timeout=3000):
                        sort_low_price.click()
                        self.logger.info("가격 낮은 순 정렬 클릭 (data-type 속성)")
                        page.wait_for_load_state('networkidle', timeout=5000)
                    else:
                        # 정렬 옵션 시도 2: 텍스트 내용 이용
                        sort_options = page.locator('//a[contains(text(), "낮은가격순")]').first
                        if sort_options.is_visible(timeout=2000):
                            sort_options.click()
                            self.logger.info("가격 낮은 순 정렬 클릭 (텍스트 기반)")
                            page.wait_for_load_state('networkidle', timeout=5000)
                        else:
                            # 정렬 옵션 시도 3: 폼 내의 select 요소 사용
                            sort_select = page.locator('select[name="sort"]').first
                            if sort_select.is_visible(timeout=2000):
                                # 낮은가격순 옵션 선택 (옵션 값은 'price_asc' 또는 유사한 값일 수 있음)
                                sort_select.select_option('price_asc')
                                self.logger.info("가격 낮은 순 정렬 선택 (select 요소)")
                                page.wait_for_load_state('networkidle', timeout=5000)
                            else:
                                self.logger.warning("가격 낮은 순 정렬 옵션을 찾을 수 없음")
                    
                    # 정렬 결과 확인용 스크린샷
                    try:
                        screenshot_path = f"screenshots/sorted_results_{query}_{int(time.time())}.png"
                        page.screenshot(path=screenshot_path)
                        self.logger.info(f"정렬 결과 스크린샷 저장: {screenshot_path}")
                    except Exception as e:
                        pass
                    
                except Exception as sort_error:
                    self.logger.error(f"가격 정렬 중 오류: {sort_error}")
                
                # 4. 첫 번째 상품 찾기
                self.logger.info("첫 번째 상품 찾기")
                
                # 상품 리스트 컨테이너 찾기
                product_list = page.locator(".product_lists .product").first
                
                if not product_list.is_visible(timeout=5000):
                    self.logger.error("상품 목록을 찾을 수 없음")
                    return None
                
                # 첫 번째 상품의 링크와 기본 정보 추출
                first_product_link = None
                
                # 상품 링크 시도 1: 이미지 링크
                img_link = product_list.locator(".pic a").first
                if img_link.is_visible(timeout=2000):
                    first_product_link = img_link.get_attribute("href")
                    self.logger.info(f"첫 번째 상품 이미지 링크: {first_product_link}")
                
                # 상품 링크 시도 2: 상품명 링크
                if not first_product_link:
                    name_link = product_list.locator(".name a").first
                    if name_link.is_visible(timeout=2000):
                        first_product_link = name_link.get_attribute("href")
                        self.logger.info(f"첫 번째 상품 이름 링크: {first_product_link}")
                
                if not first_product_link:
                    self.logger.error("첫 번째 상품의 링크를 찾을 수 없음")
                    return None
                
                # 상대 경로를 절대 경로로 변환
                if not first_product_link.startswith(("http://", "https://")):
                    first_product_link = urljoin(self.base_url, first_product_link)
                
                # 기본 정보 추출 (이름, 가격, 이미지)
                name = ""
                price = 0.0
                image_url = ""
                
                # 상품명
                name_elem = product_list.locator(".name").first
                if name_elem.is_visible(timeout=1000):
                    name = name_elem.text_content().strip()
                    
                # 가격
                price_elem = product_list.locator(".price").first
                if price_elem.is_visible(timeout=1000):
                    price_text = price_elem.text_content().strip()
                    price_match = re.search(r'[\d,]+', price_text)
                    if price_match:
                        try:
                            price = float(price_match.group().replace(",", ""))
                        except ValueError:
                            pass
                
                # 이미지
                img_elem = product_list.locator(".pic img, .img img").first
                if img_elem.is_visible(timeout=1000):
                    img_src = img_elem.get_attribute("src")
                    if img_src:
                        if not img_src.startswith(("http://", "https://")):
                            img_src = urljoin(self.base_url, img_src)
                        image_url = img_src
                
                # 상품 ID 추출
                product_id = None
                id_match = re.search(r'no=(\d+)', first_product_link)
                if id_match:
                    product_id = id_match.group(1)
                else:
                    # ID를 찾을 수 없으면 URL로 해시 생성
                    product_id = hashlib.md5(first_product_link.encode()).hexdigest()
                
                # 5. 첫 번째 상품 상세 페이지로 이동
                self.logger.info(f"첫 번째 상품 상세 페이지로 이동: {first_product_link}")
                
                try:
                    page.goto(first_product_link, wait_until='domcontentloaded', timeout=30000)
                    page.wait_for_load_state('networkidle', timeout=10000)
                    
                    # 상세 페이지 스크린샷 저장
                    try:
                        screenshot_path = f"screenshots/first_product_detail_{product_id}_{int(time.time())}.png"
                        page.screenshot(path=screenshot_path)
                        self.logger.info(f"첫 번째 상품 상세 스크린샷 저장: {screenshot_path}")
                    except Exception as e:
                        pass
                    
                    # 6. 상품 상세 정보 추출
                    item_data = {
                        "title": name,
                        "price": price,
                        "link": first_product_link,
                        "image_url": image_url,
                        "product_id": product_id
                    }
                    
                    detailed_product = self._extract_product_details(page, item_data)
                    
                    if detailed_product:
                        self.logger.info(f"최저가 상품 상세 정보 추출 성공: {detailed_product.name}")
                        return detailed_product
                    else:
                        self.logger.error("상품 상세 정보 추출 실패")
                        return None
                        
                except Exception as detail_error:
                    self.logger.error(f"상품 상세 페이지 접근 또는 정보 추출 중 오류: {detail_error}")
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
                response = page.goto(product_url, wait_until='domcontentloaded')
                
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
                    
                self.logger.error(f"상품 정보 추출 실패: {product_url}")
                return None
                
            except Exception as e:
                self.logger.error(f"상품 정보 조회 중 오류 발생: {e}", exc_info=True)
                return None

    def search_product(self, query: str, max_items: int = 50) -> List[Product]:
        """제품 검색 수행"""
        if not query:
            return []
            
        # 작업메뉴얼 기준 상품명 전처리
        query = self._preprocess_product_name(query)
        self.logger.info(f"전처리된 검색어: '{query}'")
            
        encoded_query = urllib.parse.quote(query)
        search_url = f"{self.base_url}/ez/mall.php?search_str={encoded_query}"
        
        self.logger.info(f"검색 시작: {query} (URL: {search_url})")
        
        with self.thread_context() as (browser, page):
            if not browser or not page:
                return [self._create_no_match_product(query, is_error=True)]
                
            try:
                response = page.goto(search_url, wait_until='domcontentloaded')
                if not response or not response.ok:
                    return [self._create_no_match_product(query, is_error=True)]
                
                # 검색 결과 추출
                products = self._extract_search_results(page, max_items)
                if not products:
                    return [self._create_no_match_product(query)]
                    
                return products
                
            except Exception as e:
                self.logger.error(f"검색 중 오류 발생: {e}", exc_info=True)
                return [self._create_no_match_product(query, is_error=True)]

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

    def _extract_search_results(self, page: Page, max_items: int = 50) -> List[Product]:
        """검색 결과 페이지에서 제품 정보 추출 (개선 버전)"""
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
                if container.count() > 0 and container.is_visible(timeout=1000):
                    result_container = container
                    self.logger.info(f"검색 결과 컨테이너 찾음: {selector}")
                    break
            
            # 컨테이너를 찾지 못하면 전체 페이지에서 검색
            if not result_container:
                result_container = page.locator("body")
                self.logger.info("특정 컨테이너를 찾지 못해 전체 페이지에서 검색")
            
            # 2. 제품 항목 찾기 (고려기프트 공식 패턴 기준)
            # koryoproductpage.html 분석 기반으로 최적화된 선택자
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
                count = locator.count()
                
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
                    
                    # 3.1. 상품명 (정확한 계층구조 활용)
                    title = None
                    title_selectors = [".name", "div.name", "p.name", "h3.name", ".goods_name", ".product_name"]
                    
                    for selector in title_selectors:
                        title_elem = item.locator(selector).first
                        if title_elem.is_visible(timeout=500):
                            title = title_elem.text_content(timeout=1000)
                            if title and len(title.strip()) > 0:
                                title = title.strip()
                                break
                    
                    # 제목이 없으면 다음 항목으로
                    if not title or len(title.strip()) < 2:
                        self.logger.debug(f"항목 #{i}: 제목 없음, 건너뜀")
                        continue
                    
                    # 3.2. 상품 링크 및 실제 상품 ID 추출
                    link = None
                    product_id = None
                    
                    # 항목 자체가 링크인 경우
                    if item.get_attribute("href"):
                        link = item.get_attribute("href")
                    else:
                        # 내부 링크 찾기
                        link_selectors = ["a", "div.name > a", "p.name > a", ".img > a"]
                        for selector in link_selectors:
                            link_elem = item.locator(selector).first
                            if link_elem.count() > 0:
                                href = link_elem.get_attribute("href")
                                if href and ("mall.php" in href or "goods_view" in href):
                                    link = href
                                    break
                    
                    # 링크가 없으면 다음 항목으로
                    if not link:
                        self.logger.debug(f"항목 #{i}: 링크 없음, 건너뜀")
                        continue
                    
                    # 상품 ID 추출: URL에서 'no=' 파라미터 값
                    no_match = re.search(r'no=(\d+)', link)
                    if no_match:
                        product_id = no_match.group(1)
                    
                    # 상대 URL 처리
                    if not link.startswith(("http://", "https://")):
                        link = urljoin(self.base_url, link)
                    
                    # 3.3. 가격 정보
                    price = 0
                    price_selectors = [".price", "div.price", "p.price", "strong.price", "span.price"]
                    
                    for selector in price_selectors:
                        price_elem = item.locator(selector).first
                        if price_elem.is_visible(timeout=500):
                            price_text = price_elem.text_content(timeout=1000)
                            if price_text:
                                # 숫자만 추출
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
                        if img_elem.is_visible(timeout=500):
                            src = img_elem.get_attribute("src")
                            if src:
                                # 상대 URL 처리
                                if not src.startswith(("http://", "https://")):
                                    src = urljoin(self.base_url, src)
                                image_url = src
                                break
                    
                    # 3.5. 제품 데이터 생성
                    # 해시 기반 ID와 실제 상품 ID를 둘 다 저장
                    if not product_id:
                        # no= 파라미터가 없는 경우 해시 ID 사용
                        product_id = hashlib.md5(link.encode()).hexdigest()
                    
                    product = Product(
                        id=product_id,
                        name=title,
                        price=price,
                        url=link,
                        image_url=image_url,
                        status="OK" if price > 0 else "Price Not Found"
                    )
                    
                    # 결과에 추가
                    products.append(product)
                    self.logger.debug(f"항목 #{i}: '{title}' 추출 성공")
                    
                except Exception as item_error:
                    self.logger.error(f"항목 #{i} 처리 오류: {item_error}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"검색 결과 추출 오류: {e}", exc_info=True)
        
        self.logger.info(f"최종 추출된 제품 수: {len(products)}")
        return products

    def _retry_operation(self, operation_fn, *args, max_retries=3, retry_delay=1.0, **kwargs):
        """
        작업을 지정된 횟수만큼 재시도하는 유틸리티 함수
        
        Args:
            operation_fn: 재시도할 함수
            *args: 함수에 전달할 위치 인자
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 대기 시간(초)
            **kwargs: 함수에 전달할 키워드 인자
            
        Returns:
            함수의 반환값 또는 모든 재시도가 실패하면 None
        """
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # 작업 실행
                result = operation_fn(*args, **kwargs)
                if result is not None:
                    return result
                    
                # None 결과는 실패로 간주하고 재시도
                retries += 1
                self.logger.warning(f"작업 결과가 None, 재시도 {retries}/{max_retries}")
                
            except (PlaywrightTimeoutError, PlaywrightError) as e:
                # Playwright 관련 오류
                retries += 1
                last_error = e
                self.logger.warning(f"Playwright 오류 발생, 재시도 {retries}/{max_retries}: {e}")
                
            except Exception as e:
                # 기타 예외
                retries += 1
                last_error = e
                self.logger.warning(f"예외 발생, 재시도 {retries}/{max_retries}: {e}")
            
            # 마지막 시도가 아니면 대기 후 재시도
            if retries <= max_retries:
                # 지수 백오프 적용 (재시도마다 대기 시간 증가)
                wait_time = retry_delay * (2 ** (retries - 1))
                # 무작위성 추가 (0.5배~1.5배)
                jitter = random.uniform(0.5, 1.5)
                actual_wait = wait_time * jitter
                time.sleep(actual_wait)
        
        # 모든 재시도 실패
        self.logger.error(f"최대 재시도 횟수({max_retries}회) 초과, 마지막 오류: {last_error}")
        return None
    
    def get_product_with_retry(self, product_id: str) -> Optional[Product]:
        """
        상품 ID로 상품 정보를 가져오되, 실패 시 자동 재시도
        
        Args:
            product_id: 상품 ID
            
        Returns:
            Product 객체 또는 None
        """
        return self._retry_operation(self.get_product, product_id)
    
    def search_product_with_retry(self, query: str, max_items: int = 50) -> List[Product]:
        """
        검색을 수행하되, 실패 시 자동 재시도
        
        Args:
            query: 검색어
            max_items: 최대 결과 수
            
        Returns:
            검색 결과 제품 목록
        """
        result = self._retry_operation(self.search_product, query, max_items)
        if result is None:
            # 모든 재시도 실패 시 빈 목록 대신 오류 표시 제품 반환
            return [self._create_no_match_product(query, is_error=True)]
        return result
    
    def save_results_to_excel(self, products: List[Product], filename: str = "koryo_results.xlsx"):
        """
        검색 결과를 Excel 파일로 저장
        
        Args:
            products: 저장할 제품 목록
            filename: 저장할 파일명
        """
        try:
            import pandas as pd
            
            # 결과가 없으면 빈 파일 생성하지 않음
            if not products:
                self.logger.warning("저장할 제품 정보가 없습니다.")
                return
            
            # 제품 데이터를 딕셔너리 목록으로 변환
            data = []
            for product in products:
                # 기본 정보
                product_dict = {
                    "상품명": product.name,
                    "가격": product.price,
                    "URL": product.url,
                    "이미지URL": product.image_url,
                    "상품코드": product.product_code or "",
                    "상태": product.status or "",
                    "소스": product.source or "koryo"
                }
                
                # 수량별 가격 정보
                if product.quantity_prices:
                    for qty, price in product.quantity_prices.items():
                        product_dict[f"수량{qty}개_가격"] = price
                
                # 사양 정보
                if product.specifications:
                    for key, value in product.specifications.items():
                        # 열 이름 충돌 방지
                        product_dict[f"사양_{key}"] = value
                
                data.append(product_dict)
            
            # DataFrame 생성 및 파일 저장
            df = pd.DataFrame(data)
            output_path = os.path.join("output", filename)
            df.to_excel(output_path, index=False)
            
            self.logger.info(f"제품 정보 {len(products)}개를 '{output_path}'에 저장했습니다.")
            
        except ImportError:
            self.logger.error("pandas 모듈이 설치되어 있지 않아 Excel 저장이 불가능합니다.")
        except Exception as e:
            self.logger.error(f"Excel 파일 저장 중 오류 발생: {e}", exc_info=True)

    def search_tumbler(self, max_items: int = 50) -> List[Product]:
        """텀블러 전용 검색 메서드 (koryoaftersearch.html 참고)"""
        self.logger.info("텀블러 검색 시작")
        
        # 텀블러 카테고리 URL (실제 작동하는 URL)
        search_url = f"{self.base_url}/ez/mall.php?cat=013001000"
        
        with self.thread_context() as (browser, page):
            if not browser or not page:
                self.logger.error("브라우저 컨텍스트 생성 실패")
                return []
            
            try:
                # 페이지 이동
                self.logger.info(f"텀블러 카테고리 페이지 접속: {search_url}")
                response = page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                
                if not response or not response.ok:
                    self.logger.error(f"페이지 접속 실패: {search_url}")
                    return []
                
                # 텀블러 필터링 (카테고리 내에서 '텀블러' 검색)
                try:
                    # 결과 내 검색 입력창 찾기
                    search_input = page.locator("#re_keyword").first
                    if search_input.count() > 0:
                        search_input.fill("텀블러")
                        
                        # 검색 버튼 클릭
                        search_button = page.locator('button[onclick="re_search()"]').first
                        if search_button.count() > 0:
                            search_button.click()
                            page.wait_for_load_state('networkidle', timeout=10000)
                    else:
                        self.logger.warning("결과 내 검색 입력창을 찾을 수 없음")
                except Exception as e:
                    self.logger.warning(f"결과 내 검색 실패: {e}")
                
                # 스크린샷 저장 (디버깅용)
                try:
                    os.makedirs("screenshots", exist_ok=True)
                    screenshot_path = f"screenshots/tumbler_search_{int(time.time())}.png"
                    page.screenshot(path=screenshot_path)
                    self.logger.info(f"스크린샷 저장: {screenshot_path}")
                except Exception as e:
                    self.logger.debug(f"스크린샷 저장 실패: {e}")
                
                # 제품 목록 추출 (koryoaftersearch.html 기반 최적화)
                products = self._extract_tumbler_results(page, max_items)
                
                if products:
                    self.logger.info(f"텀블러 검색 결과 {len(products)}개 추출 성공")
                    return products
                else:
                    self.logger.warning("텀블러 검색 결과 없음")
                    return []
                
            except Exception as e:
                self.logger.error(f"텀블러 검색 중 오류 발생: {e}", exc_info=True)
                return []
    
    def _extract_tumbler_results(self, page: Page, max_items: int = 50) -> List[Product]:
        """텀블러 검색 결과 페이지에서 제품 정보 추출 (koryoaftersearch.html 최적화)"""
        products = []
        try:
            # 1. 제품 총 개수 확인
            try:
                count_text = page.locator('.list_info > div > span').first.text_content()
                if count_text:
                    self.logger.info(f"검색 결과 총 개수: {count_text}")
            except Exception as e:
                self.logger.debug(f"총 개수 확인 실패: {e}")
            
            # 2. 제품 목록 컨테이너 찾기 (koryoaftersearch.html 기반)
            container = page.locator('.product_lists').first
            if not container or container.count() == 0:
                self.logger.warning("제품 목록 컨테이너를 찾을 수 없음")
                return []
            
            # 3. 개별 제품 추출
            items = container.locator('.product')
            count = items.count()
            
            if count == 0:
                self.logger.warning("제품 항목을 찾을 수 없음")
                return []
            
            self.logger.info(f"제품 항목 {count}개 발견")
            
            # 4. 각 제품 정보 추출
            for i in range(min(count, max_items)):
                try:
                    item = items.nth(i)
                    
                    # 4.1. 상품명 (div.name > a)
                    name_elem = item.locator('div.name > a').first
                    if not name_elem.count() > 0:
                        self.logger.debug(f"항목 #{i}: 상품명 요소 없음")
                        continue
                    
                    title = name_elem.text_content()
                    if not title:
                        self.logger.debug(f"항목 #{i}: 상품명 텍스트 없음")
                        continue
                    
                    # 4.2. 링크 (div.name > a[href])
                    link = name_elem.get_attribute('href')
                    if not link:
                        self.logger.debug(f"항목 #{i}: 링크 없음")
                        continue
                    
                    # URL이 상대 경로면 절대 경로로 변환
                    if not link.startswith(('http://', 'https://')):
                        link = urljoin(self.base_url, link)
                    
                    # 4.3. 상품 ID 추출 (no= 파라미터)
                    product_id = None
                    no_match = re.search(r'no=(\d+)', link)
                    if no_match:
                        product_id = no_match.group(1)
                    else:
                        # ID가 없으면 해시 값 사용
                        product_id = hashlib.md5(link.encode()).hexdigest()
                    
                    # 4.4. 모델 번호 (div.model)
                    model_elem = item.locator('div.model').first
                    model_code = ""
                    if model_elem.count() > 0:
                        model_code = model_elem.text_content().strip()
                    
                    # 4.5. 가격 (div.price)
                    price_elem = item.locator('div.price').first
                    price = 0
                    if price_elem.count() > 0:
                        price_text = price_elem.text_content().strip()
                        price_match = re.search(r'[\d,]+', price_text)
                        if price_match:
                            try:
                                price = float(price_match.group().replace(',', ''))
                            except ValueError:
                                pass
                    
                    # 4.6. 이미지 (div.pic > a > img)
                    img_elem = item.locator('div.pic > a > img').first
                    image_url = None
                    if img_elem.count() > 0:
                        image_url = img_elem.get_attribute('src')
                        if image_url and not image_url.startswith(('http://', 'https://')):
                            image_url = urljoin(self.base_url, image_url)
                    
                    # 실제 이미지 URL이 없으면 패턴 기반으로 생성
                    if not image_url or "ico_shop.gif" in image_url:
                        image_url = f"{self.base_url}/ez/upload/mall/shop_{product_id}_0.jpg"
                    
                    # 4.7. 제품 객체 생성
                    product = Product(
                        id=product_id,
                        name=title.strip(),
                        price=price,
                        url=link,
                        image_url=image_url,
                        source="koryo",
                        product_code=model_code,
                        status="OK" if price > 0 else "Price Not Found"
                    )
                    
                    # 결과에 추가
                    products.append(product)
                    self.logger.debug(f"항목 #{i}: '{title.strip()}' 추출 성공")
                    
                except Exception as item_error:
                    self.logger.error(f"항목 #{i} 처리 오류: {item_error}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"텀블러 결과 추출 오류: {e}", exc_info=True)
        
        self.logger.info(f"최종 추출된 텀블러 제품 수: {len(products)}")
        return products
        
    def get_tumbler_detail(self, product_id: str) -> Optional[Product]:
        """텀블러 상세 정보 조회 (koryoproductpage.html 최적화)"""
        self.logger.info(f"텀블러 상세 정보 조회 시작: {product_id}")
        
        # 상세 페이지 URL
        detail_url = f"{self.base_url}/ez/mall.php?query=view&no={product_id}"
        
        with self.thread_context() as (browser, page):
            if not browser or not page:
                self.logger.error(f"브라우저 컨텍스트 생성 실패: {product_id}")
                return None
            
            try:
                # 페이지 접속
                self.logger.info(f"상세 페이지 접속: {detail_url}")
                response = page.goto(detail_url, wait_until='domcontentloaded', timeout=30000)
                
                if not response or not response.ok:
                    self.logger.error(f"상세 페이지 접속 실패: {detail_url}")
                    return None
                
                # 스크린샷 저장 (디버깅용)
                try:
                    os.makedirs("screenshots", exist_ok=True)
                    screenshot_path = f"screenshots/tumbler_detail_{product_id}_{int(time.time())}.png"
                    page.screenshot(path=screenshot_path)
                    self.logger.info(f"스크린샷 저장: {screenshot_path}")
                except Exception as e:
                    self.logger.debug(f"스크린샷 저장 실패: {e}")
                
                # 제품 상세 정보 추출
                try:
                    # 1. 상품명
                    name = ""
                    name_elem = page.locator('.product_name').first
                    if name_elem.count() > 0:
                        name = name_elem.text_content().strip()
                    
                    if not name:
                        self.logger.error("상품명을 찾을 수 없음")
                        return None
                    
                    # 2. 상품 코드
                    product_code = ""
                    code_elems = [
                        page.locator('//div[contains(text(), "상품코드")]').first,
                        page.locator('//td[contains(text(), "상품코드")]').first
                    ]
                    
                    for elem in code_elems:
                        if elem.count() > 0:
                            code_text = elem.text_content().strip()
                            code_match = re.search(r'상품코드\s*[:\-]?\s*([A-Za-z0-9_-]+)', code_text)
                            if code_match:
                                product_code = code_match.group(1)
                                break
                    
                    # 3. 가격
                    price = 0
                    price_elem = page.locator('#main_price').first
                    if price_elem.count() > 0:
                        price_text = price_elem.text_content().strip()
                        price_match = re.search(r'[\d,]+', price_text)
                        if price_match:
                            try:
                                price = float(price_match.group().replace(',', ''))
                            except ValueError:
                                pass
                    
                    # 4. 수량별 가격
                    quantity_prices = {}
                    qty_table = page.locator('table.quantity_price__table')
                    
                    if qty_table.count() > 0:
                        # 첫 번째 행: 수량
                        qty_cells = qty_table.locator('tr').first.locator('td').all()
                        # 두 번째 행: 가격
                        price_cells = qty_table.locator('tr').nth(1).locator('td').all()
                        
                        for i in range(min(len(qty_cells), len(price_cells))):
                            qty_text = qty_cells[i].text_content().strip()
                            price_text = price_cells[i].text_content().strip()
                            
                            qty_match = re.search(r'(\d[\d,]*)', qty_text)
                            price_match = re.search(r'[\d,]+', price_text)
                            
                            if qty_match and price_match:
                                try:
                                    qty = int(qty_match.group(1).replace(',', ''))
                                    unit_price = float(price_match.group().replace(',', ''))
                                    quantity_prices[str(qty)] = unit_price
                                except ValueError:
                                    pass
                    
                    # 5. 이미지
                    # 메인 이미지
                    main_image_url = None
                    main_img = page.locator('#main_img')
                    if main_img.count() > 0:
                        src = main_img.get_attribute('src')
                        if src:
                            if not src.startswith(('http://', 'https://')):
                                src = urljoin(self.base_url, src)
                            main_image_url = src
                    
                    # 이미지 없으면 패턴 기반으로 생성
                    if not main_image_url:
                        main_image_url = f"{self.base_url}/ez/upload/mall/shop_{product_id}_0.jpg"
                    
                    # 썸네일 이미지
                    image_gallery = []
                    thumbs = page.locator('.product_picture .thumnails img').all()
                    
                    for thumb in thumbs:
                        src = thumb.get_attribute('src')
                        if src and not self._is_ui_image(src):
                            if not src.startswith(('http://', 'https://')):
                                src = urljoin(self.base_url, src)
                            if src not in image_gallery:
                                image_gallery.append(src)
                    
                    # 6. 사양
                    specifications = {}
                    specs_table = page.locator('table.tbl_info')
                    
                    if specs_table.count() > 0:
                        rows = specs_table.locator('tr').all()
                        
                        for row in rows:
                            th = row.locator('th').first
                            td = row.locator('td').first
                            
                            if th.count() > 0 and td.count() > 0:
                                key = th.text_content().strip().rstrip(':')
                                value = td.text_content().strip()
                                
                                if key and value:
                                    specifications[key] = value
                
                    # 7. 상세 설명
                    description = None
                    desc_elem = page.locator('div.prd_detail')
                    
                    if desc_elem.count() > 0:
                        try:
                            html = desc_elem.inner_html()
                            if html and len(html) > 50:
                                description = html
                        except Exception as desc_error:
                            self.logger.debug(f"상세 설명 HTML 추출 실패: {desc_error}")
                            text = desc_elem.text_content()
                            if text and len(text) > 50:
                                description = text
                    
                    # 8. 제품 객체 생성
                    product = Product(
                        id=product_id,
                        name=name,
                        price=price,
                        url=detail_url,
                        image_url=main_image_url,
                        image_gallery=image_gallery if image_gallery else None,
                        product_code=product_code,
                        quantity_prices=quantity_prices if quantity_prices else None,
                        specifications=specifications if specifications else None,
                        description=description,
                        source="koryo",
                        status="OK" if price > 0 else "Price Not Found"
                    )
                    
                    self.logger.info(f"텀블러 상세 정보 추출 성공: {name}")
                    return product
                    
                except Exception as extract_error:
                    self.logger.error(f"텀블러 상세 정보 추출 실패: {extract_error}", exc_info=True)
                    return None
                
            except Exception as e:
                self.logger.error(f"텀블러 상세 정보 조회 중 오류 발생: {e}", exc_info=True)
                return None

    def full_search_flow(self, query: str, max_items: int = 10) -> List[Product]:
        """
        완전한 검색 흐름 구현: 메인 페이지 -> 검색 -> 결과 목록 -> 상세 정보
        
        Args:
            query: 검색어
            max_items: 최대 결과 수
            
        Returns:
            Product 객체 리스트 (상세 정보 포함)
        """
        self.logger.info(f"전체 검색 흐름 시작: '{query}'")
        
        detailed_products = []
        
        with self.thread_context() as (browser, page):
            if not browser or not page:
                self.logger.error("브라우저 컨텍스트 생성 실패")
                return []
            
            try:
                # 1. 메인 페이지 접속
                self.logger.info(f"메인 페이지 접속: {self.base_url}")
                response = page.goto(self.base_url, wait_until='domcontentloaded', timeout=30000)
                
                if not response or not response.ok:
                    self.logger.error(f"메인 페이지 접속 실패: {self.base_url}")
                    return []
                
                # 페이지 로딩 대기 (네트워크 활동 종료 대기)
                self._wait_for_load_state(page)
                
                # 2. 검색 수행
                try:
                    # 검색창 찾기 (메인페이지 검색)
                    search_input = page.locator("#main_keyword").first
                    if search_input.is_visible(timeout=5000):
                        # 검색어 입력
                        search_input.fill(query)
                        self.logger.info(f"검색어 입력: '{query}'")
                        
                        # 검색 버튼 클릭
                        search_button = page.locator(".search_btn_div img").first
                        if search_button.is_visible(timeout=3000):
                            search_button.click()
                            self.logger.info("검색 버튼 클릭")
                            # 페이지 로딩 대기
                            page.wait_for_load_state('networkidle', timeout=15000)
                        else:
                            # 엔터키 입력으로 검색
                            search_input.press("Enter")
                            self.logger.info("엔터키로 검색 제출")
                            page.wait_for_load_state('networkidle', timeout=15000)
                    else:
                        # 검색창을 찾을 수 없는 경우 직접 URL 이동
                        self.logger.warning("메인 검색창을 찾을 수 없어 직접 URL 이동")
                        encoded_query = urllib.parse.quote(query)
                        search_url = f"{self.base_url}/ez/mall.php?search_str={encoded_query}"
                        page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                        page.wait_for_load_state('networkidle', timeout=15000)
                except Exception as search_error:
                    self.logger.error(f"메인 페이지 검색 중 오류: {search_error}")
                    # 직접 URL 이동으로 대체
                    encoded_query = urllib.parse.quote(query)
                    search_url = f"{self.base_url}/ez/mall.php?search_str={encoded_query}"
                    page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
                
                # 스크린샷 저장 (디버깅용)
                try:
                    os.makedirs("screenshots", exist_ok=True)
                    screenshot_path = f"screenshots/search_results_{query}_{int(time.time())}.png"
                    page.screenshot(path=screenshot_path)
                    self.logger.info(f"검색 결과 스크린샷 저장: {screenshot_path}")
                except Exception as e:
                    self.logger.debug(f"스크린샷 저장 실패: {e}")
                
                # 3. 검색 결과 추출
                products = self._extract_search_results(page, max_items)
                if not products:
                    self.logger.warning(f"'{query}' 검색 결과가 없습니다.")
                    return []
                
                self.logger.info(f"추출된 검색 결과: {len(products)}개")
                
                # 4. 각 제품의 상세 정보 조회
                for i, product in enumerate(products[:min(max_items, len(products))]):
                    try:
                        self.logger.info(f"상품 {i+1}/{len(products)}: '{product.name}' 상세 정보 조회")
                        
                        # 상세 페이지 URL
                        if not product.url:
                            self.logger.warning(f"상품 URL이 없어 건너뜀: {product.name}")
                            continue
                        
                        # 상세 페이지 이동
                        page.goto(product.url, wait_until='domcontentloaded', timeout=30000)
                        page.wait_for_load_state('networkidle', timeout=10000)
                        
                        # 스크린샷 저장 (디버깅용)
                        try:
                            screenshot_path = f"screenshots/product_detail_{product.id}_{int(time.time())}.png"
                            page.screenshot(path=screenshot_path)
                            self.logger.debug(f"상품 상세 스크린샷 저장: {screenshot_path}")
                        except Exception as e:
                            pass
                        
                        # 상세 정보 추출
                        item_data = {
                            "title": product.name,
                            "price": product.price,
                            "link": product.url,
                            "image_url": product.image_url,
                            "product_id": product.id
                        }
                        
                        detailed_product = self._extract_product_details(page, item_data)
                        if detailed_product:
                            self.logger.info(f"상품 상세 정보 추출 성공: {detailed_product.name}")
                            detailed_products.append(detailed_product)
                        else:
                            self.logger.warning(f"상품 상세 정보 추출 실패: {product.name}")
                            # 기본 정보만 포함된 제품 추가
                            detailed_products.append(product)
                        
                        # 페이지 사이 지연 시간
                        time.sleep(1)
                        
                    except Exception as detail_error:
                        self.logger.error(f"상품 {product.name} 상세 정보 조회 중 오류: {detail_error}")
                        # 기본 정보만 포함된 제품 추가
                        detailed_products.append(product)
                
                self.logger.info(f"'{query}' 검색 완료: 총 {len(detailed_products)}개 상품 상세 정보 추출")
                return detailed_products
                
            except Exception as e:
                self.logger.error(f"전체 검색 흐름 중 오류 발생: {e}", exc_info=True)
                return []

        