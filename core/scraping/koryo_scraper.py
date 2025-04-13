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
from typing import Any, Dict, List, Optional, Tuple
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
        # BaseMultiLayerScraper 상속을 위한 파라미터 (Playwright 설정으로 대체)
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        # Playwright 설정을 ScraperConfig에서 가져옴
        self.config = config or ScraperConfig()

        # BaseScraper 초기화 (max_retries, timeout은 Playwright 설정 사용)
        # FileCache 사용을 위해 cache 전달
        super().__init__(max_retries=self.config.max_retries, timeout=self.config.timeout, cache=cache)

        self.logger = logging.getLogger(__name__)
        self.base_url = "https://koreagift.com"
        self.mall_url = f"{self.base_url}/ez/mall.php"
        self.search_url = f"{self.base_url}/ez/goods/goods_search.php"

        # Playwright 인스턴스 및 리소스 (초기화는 init_playwright에서 수행)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

        # Thread-local storage for browser contexts
        self._thread_local = threading.local()

        # --- 셀렉터 및 패턴 정의 --- (기존 구조 유지, 필요시 수정)
        self.title_selectors = [
            ".product_name", # product page analysis
            "h3.detail_tit",
            "h3.prd_title",
            ".view_title",
            "h1.product-title",
            ".name", # main page best100 analysis
            "h2.name",
            ".goods_name",
            "div.title",
            "div.product-title",
            "div.prd-name"
        ]
        self.price_selectors = [
            "#main_price", # product page analysis
            "dl.detail_price dd",
            ".price_num",
            ".product-price",
            ".price", # main page best100 analysis
            ".product-price-num",
            "div.price",
            "strong.price",
            ".goods_price",
            "p.price"
        ]
        self.code_selectors = [
            '//div[contains(text(), "상품코드")]', # product page analysis (XPath needed for text containment)
            # "div.product_code", # Original, less specific
            ".product-code",
            ".item-code",
            "span.code",
            'td:contains("상품코드") + td',
            'li:contains("상품코드")',
            '.goods_code',
            '//span[contains(text(), "상품코드")]',
            '//td[contains(text(), "상품코드")]'
        ]
        self.image_selectors = [
            "#main_img", # product page analysis
            ".product_picture .thumnails img", # product page analysis
            "div.swiper-slide img",
            ".product-image img",
            ".detail_img img",
            ".product-gallery img",
            # "#main_img", # Duplicate, covered above
            # ".product_picture .thumnails img", # Duplicate, covered above
            ".tbl_info img", # product page analysis (specs table images)
            "div.prd_detail img",
            ".goods_image img",
            ".prd-img img",
            "div.thumb img",
            ".carousel-inner img",
            "div.pic img"
        ]
        self.selectors = {
            "search_results_container": {  # 여러 선택자 추가
                "selector": "ul.prd_list, div.prd_list_wrap, table.mall_list, div.product_list, .prd-list, .best100_tab, .goods-list, .product-grid"
            },
            "search_result_item": {  # 여러 선택자 추가
                "selector": "li.prd, ul.prd_list li, table.mall_list td.prd, div.product_list .item, .prd-list .prd-item, .product, .best100_tab .product, .goods-item, .product-grid-item"
            },
            "product_list": {
                # Priority to more specific structures if known context (e.g., best100)
                # General selectors first for broader compatibility
                "selector": "div.prd_list_wrap li.prd, ul.prd_list li, table.mall_list td.prd, div.product_list .item, .prd-list .prd-item, .product, .best100_tab .product, .goods-list li, .product-grid .item", # crowling_kogift.py .product 추가, main page best100 analysis 추가
            },
            "product_title_list": { # 목록용 제목
                 "selector": ".name, p.name a, div.name a, td.name a, .prd_name a, a.product-title, div.name > a, .goods-name a, .product-name a, h3.name a", # main page best100 analysis 추가, crowling_kogift.py div.name > a 추가
            },
            "product_link_list": { # 목록용 링크
                 "selector": "a, p.name a, div.name a, td.name a, .prd_name a, a.product-title, div.prd_list_wrap li.prd > a, div.pic > a, .goods-item > a, .product-link", # Ensure 'a' itself is included for cases like '.product a', crowling_kogift.py div.pic > a 추가
                 "attribute": "href",
            },
            "price_list": { # 목록용 가격
                "selector": ".price, p.price, div.price, td.price, .prd_price, span.price, strong.price, .goods-price, .product-price", # main page best100 analysis 추가, crowling_kogift.py div.price 추가
            },
            "thumbnail_list": { # 목록용 썸네일
                "selector": ".img img, .pic img, img.prd_img, td.img img, .thumb img, img.product-image, div.pic > a > img, .goods-thumb img, .product-thumb img", # main page best100 analysis 추가, crowling_kogift.py div.pic > a > img 추가
                "attribute": "src"
            },
            "next_page_js": { # JavaScript 기반 페이징 ('다음' 또는 페이지 번호)
                 "selector": '#pageindex a[onclick*="getPageGo"], div.custom_paging div[onclick*="getPageGo"], .paging a.next, a:contains("다음")', # product page review pagination analysis 추가
            },
            "next_page_href": { # href 기반 페이징
                 # Consider specific pagination structures if known (e.g., #pageindex for reviews)
                 "selector": '#pageindex a:not([onclick]), .paging a.next, a.next[href]:not([href="#"]):not([href^="javascript:"]), a:contains("다음")[href]:not([href="#"]):not([href^="javascript:"])', # product page review pagination analysis 추가, more specific href check
                 "attribute": "href",
            },
            "quantity_table": {
                "selector": "table.quantity_price__table", # product page analysis
            },
            "specs_table": {
                "selector": "table.tbl_info", # product page analysis
            },
            "description": {
                # product page analysis mentions #prd_detail_content, but also .prd_detail contains the table etc.
                # Might need refinement based on what part is desired (text vs html, full vs partial)
                "selector": "div.prd_detail, #prd_detail_content",
            },
            "category_items": {
                 # main page analysis mentions #div_category_all .tc_link
                "selector": "#div_category_all .tc_link, #lnb_menu > li > a, .category a, #category_all a, .menu_box a, a[href*='mall.php?cat='], a[href*='mall.php?cate=']", # cate 추가
            },
            # --- crowling_kogift.py 에서 사용된 셀렉터들 --- 
            "main_search_input": 'input[name="keyword"][id="main_keyword"]', # main page analysis confirms id
            "main_search_button": 'img#search_submit',
            "re_search_input": 'input#re_keyword',
            "re_search_button": 'button[onclick="re_search()"]',
            "product_count": '//div[contains(text(), "개의 상품이 있습니다.")]/span', # XPath 사용 예시
            "model_text": { # Model / Product code on list/search page
                "selector": ".product .model"
            },
            "options_select": { # Product page options analysis
                "selector": 'select[name^="option_"]'
            },
            "review_table": { # Product page reviews analysis
                 "selector": 'table.tbl_review'
            },
            "review_pagination": { # Product page reviews analysis
                 "selector": '#pageindex'
            }
        }
        self.patterns = {
            "price_number": re.compile(r"[\d,]+"),
            "product_code": re.compile(r"상품코드\s*[:\-]?\s*([A-Za-z0-9_-]+)"),
            "quantity": re.compile(r"(\d+)\s*(?:개|세트|묶음|ea)", re.IGNORECASE),
            "quantity_price": re.compile(r"(\d+)개[:\s]+([0-9,]+)원"),
            "vat_included": re.compile(r"VAT\s*(포함|별도|제외)", re.IGNORECASE),
            "js_page_number": re.compile(r"getPageGo\d*\((\d+)\)"), # JS 페이징 번호 추출
        }
        self.default_categories = [
            ("볼펜/사무용품", "013001000"),
            ("텀블러/머그컵", "013002000"),
            ("가방", "013003000"),
            ("전자/디지털", "013004000"),
        ]
        os.makedirs("output", exist_ok=True)

        # Set base URLs to try
        self.base_search_urls = [
            "https://www.koreagift.com/ez/goods/goods_search.php",  # www 서브도메인을 첫번째로
            "https://koreagift.com/ez/goods/goods_search.php",
            "https://www.adpanchok.co.kr/ez/goods/goods_search.php",  # www 서브도메인 추가
            "https://adpanchok.co.kr/ez/goods/goods_search.php",
            "https://www.koreagift.com/ez/index.php",  # www 서브도메인 추가
            "https://koreagift.com/ez/index.php",  # 리다이렉션 URL
            "https://www.koreagift.com",  # 루트 도메인 추가
            "https://koreagift.com"  # 루트 도메인 추가
        ]
        # Keep the original search_url for compatibility if needed elsewhere, but prefer base_search_urls for searching
        self.search_url = self.base_search_urls[0]

    # --- Playwright 초기화 및 종료 --- 

    def init_playwright(self) -> bool:
        """Playwright 인스턴스 및 브라우저 초기화"""
        if self._page and not self._page.is_closed():
            self.logger.debug("Playwright page already initialized and open.")
            return True
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.config.headless,
                args=['--disable-web-security', '--no-sandbox', '--disable-features=IsolateOrigins,site-per-process']
            )
            context = self._browser.new_context(
                user_agent=self.config.user_agent,
                viewport={'width': 1920, 'height': 1080}, # 필요시 뷰포트 설정
                ignore_https_errors=True  # SSL 인증서 오류 무시
            )
            # 기본 타임아웃 설정 (늘림)
            context.set_default_timeout(self.config.timeout * 2)
            context.set_default_navigation_timeout(self.config.navigation_timeout * 2)

            self._page = context.new_page()
            self.logger.info("Playwright initialized successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Playwright: {e}", exc_info=True)
            self.close() # 실패 시 자원 정리
            return False

    def _create_new_context(self) -> tuple[Optional[Browser], Optional[Page]]:
        """Create a new browser context and page for thread-safe operations"""
        try:
            # Start a fresh playwright instance
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(
                headless=self.config.headless,
                args=[
                    '--disable-web-security', 
                    '--no-sandbox', 
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-dev-shm-usage'  # 메모리 관련 문제 방지
                ]
            )
            context = browser.new_context(
                user_agent=self.config.user_agent,
                viewport={'width': 1920, 'height': 1080},
                ignore_https_errors=True,  # SSL 인증서 오류 무시
                proxy={  # 프록시 설정 추가
                    "server": "system",  # 시스템 프록시 사용
                }
            )
            
            # Set timeouts (더 짧게 조정)
            context.set_default_timeout(self.config.timeout)
            context.set_default_navigation_timeout(self.config.navigation_timeout)
            
            # Create a new page
            page = context.new_page()
            
            # Store these in thread local storage
            self._thread_local.playwright = playwright
            self._thread_local.browser = browser
            self._thread_local.context = context
            self._thread_local.page = page
            
            return browser, page
        except Exception as e:
            self.logger.error(f"Failed to create new browser context: {e}", exc_info=True)
            self._close_thread_context()
            return None, None
            
    def _close_thread_context(self):
        """Close and clean up thread-local browser context"""
        try:
            if hasattr(self._thread_local, 'page') and self._thread_local.page:
                try:
                    self._thread_local.page.close()
                except:
                    pass
                self._thread_local.page = None
                
            if hasattr(self._thread_local, 'context') and self._thread_local.context:
                try:
                    self._thread_local.context.close()
                except:
                    pass
                self._thread_local.context = None
                
            if hasattr(self._thread_local, 'browser') and self._thread_local.browser and self._thread_local.browser.is_connected():
                try:
                    self._thread_local.browser.close()
                except:
                    pass
                self._thread_local.browser = None
                
            if hasattr(self._thread_local, 'playwright') and self._thread_local.playwright:
                try:
                    self._thread_local.playwright.stop()
                except:
                    pass
                self._thread_local.playwright = None
        except Exception as e:
            self.logger.error(f"Error cleaning up thread context: {e}", exc_info=True)

    def close(self):
        """Playwright 관련 자원 종료"""
        if self._browser and self._browser.is_connected():
            try:
                self._browser.close()
                self.logger.info("Playwright browser closed.")
            except Exception as e:
                 self.logger.error(f"Error closing Playwright browser: {e}")
        if self._playwright:
            try:
                self._playwright.stop()
                self.logger.info("Playwright instance stopped.")
            except Exception as e:
                 self.logger.error(f"Error stopping Playwright: {e}")

        self._page = None
        self._browser = None
        self._playwright = None
        
        # Also close any thread-local contexts
        self._close_thread_context()

    def __del__(self):
        """소멸자: 자원 정리 시도"""
        self.close()

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
            return locator.text_content(timeout=timeout or self.config.wait_timeout)
        except (PlaywrightTimeoutError, PlaywrightError) as e:
            self.logger.debug(f"Timeout or error getting text content: {e}")
            return None

    def _safe_get_attribute(self, locator: Locator, attribute: str, timeout: Optional[int] = None) -> Optional[str]:
        """타임아웃 및 오류 처리하여 Locator의 속성 값 가져오기"""
        try:
            return locator.get_attribute(attribute, timeout=timeout or self.config.wait_timeout)
        except (PlaywrightTimeoutError, PlaywrightError) as e:
            self.logger.debug(f"Timeout or error getting attribute '{attribute}': {e}")
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
            # 제품명
            title_locator = element_locator.locator(self.selectors["product_title_list"]["selector"]).first
            product_data["title"] = self._safe_get_text(title_locator)
            if not product_data["title"]:
                 self.logger.debug("Could not extract title from list item.")
                 return None # 제목 없으면 유효하지 않음

            # 링크
            link_locator = element_locator.locator(self.selectors["product_link_list"]["selector"]).first
            link_attr = self.selectors["product_link_list"].get("attribute", "href")
            link_href = self._safe_get_attribute(link_locator, link_attr)
            if not link_href:
                self.logger.debug("Could not extract link href from list item.")
                # 링크 없으면 상세 정보 접근 불가, 경우에 따라 처리
                # return None 
            product_data["link"] = urljoin(self.base_url, link_href.strip() if link_href else "")

            # 가격
            price_locator = element_locator.locator(self.selectors["price_list"]["selector"]).first
            price_text = self._safe_get_text(price_locator)
            price = 0.0
            if price_text:
                price_match = self.patterns["price_number"].search(price_text)
                if price_match:
                    try:
                        price = float(price_match.group().replace(",", ""))
                    except ValueError:
                         self.logger.debug(f"Could not parse price text: {price_text}")
            product_data["price"] = price

            # 모델명 추출
            model_locator = element_locator.locator(self.selectors["model_text"]["selector"]).first
            model_text = self._safe_get_text(model_locator)
            product_data["model"] = model_text.strip() if model_text else None

            # 썸네일
            thumb_locator = element_locator.locator(self.selectors["thumbnail_list"]["selector"]).first
            thumb_attr = self.selectors["thumbnail_list"].get("attribute", "src")
            thumb_src = self._safe_get_attribute(thumb_locator, thumb_attr)
            product_data["image"] = urljoin(self.base_url, thumb_src.strip()) if thumb_src else None

            # ID (링크 기반)
            if product_data["link"]:
                product_data["product_id"] = hashlib.md5(product_data["link"].encode()).hexdigest()
            else:
                 product_data["product_id"] = None # 링크 없으면 ID 생성 불가

            return product_data

        except Exception as e:
            self.logger.warning(f"Error extracting list item details: {e}", exc_info=self.config.debug)
            return None

    def _extract_product_details(self, page: Page, item: Dict) -> Optional[Product]:
        """상세 페이지(Page)에서 정보 추출 (Playwright Page 객체 사용)"""
        product_id = item.get("product_id")
        product_link = item.get("link")
        status = "Extraction Failed"  # Default status

        if not product_link:
             self.logger.warning("Missing product link in item data for detail extraction.")
             return None
        if not product_id:
            product_id = hashlib.md5(product_link.encode()).hexdigest()

        # Wait for essential elements to likely be present after navigation
        self._wait_for_selector(page, self.title_selectors[0], timeout=10000) # Wait for product name

        try:
            # 상품명 (상세 페이지 우선)
            title = item.get("title", "Unknown Product") # Use list title as fallback
            found_title = False
            for selector in self.title_selectors:
                 title_locator = page.locator(selector).first
                 # Check visibility quickly before attempting to get text
                 if title_locator.is_visible(timeout=1000):
                     detail_title = self._safe_get_text(title_locator, timeout=5000)
                     if detail_title and detail_title.strip():
                         title = detail_title.strip()
                         found_title = True
                         self.logger.debug(f"Found detail title: '{title}' using selector: {selector}")
                         break
            if not found_title:
                 self.logger.warning(f"Could not find detail title using selectors on {product_link}. Using list title: {title}")

            # 가격 (상세 페이지 우선)
            price = item.get("price", 0.0) # Use list price as fallback
            found_price = False
            for selector in self.price_selectors:
                price_locator = page.locator(selector).first
                if price_locator.is_visible(timeout=1000):
                    price_text = self._safe_get_text(price_locator, timeout=5000)
                    if price_text:
                        price_match = self.patterns["price_number"].search(price_text)
                        if price_match:
                            try:
                                price = float(price_match.group().replace(",", ""))
                                found_price = True
                                self.logger.debug(f"Found detail price: {price} using selector: {selector}")
                                break
                            except ValueError:
                                self.logger.warning(f"Could not convert detail price text '{price_match.group()}' to float.")
            if not found_price:
                self.logger.warning(f"Could not find detail price using selectors on {product_link}. Using list price: {price}")

            # If core info like title or price is missing, we might consider it a failed extraction early
            if not found_title and title == "Unknown Product":
                 self.logger.error(f"Extraction failed: Could not determine product title for {product_link}.")
                 # Optionally set status = "Title Not Found" or similar? For now, stick to generic failure.
                 return None # Cannot proceed without a title

            # 상품 코드
            product_code = None
            found_code = False
            for selector in self.code_selectors:
                # Handle XPath selector differently
                if selector.startswith("//"):
                    code_locator = page.locator(selector).first
                else:
                    code_locator = page.locator(selector).first

                # Check visibility before getting text
                if code_locator.is_visible(timeout=1000):
                     code_text = self._safe_get_text(code_locator, timeout=5000)
                     if code_text:
                         # If using the XPath selector, the text might contain "상품코드:"
                         if "상품코드" in code_text:
                              code_match = self.patterns["product_code"].search(code_text)
                              if code_match:
                                   product_code = code_match.group(1).strip()
                                   found_code = True
                                   self.logger.debug(f"Found product code: '{product_code}' using selector: {selector} (pattern match)")
                                   break
                         # For other selectors, or if pattern fails, use the text directly if it looks like a code
                         elif len(code_text.strip()) > 3 and not code_text.strip().startswith("상품"):
                              product_code = code_text.strip()
                              found_code = True
                              self.logger.debug(f"Found product code: '{product_code}' using selector: {selector} (direct text)")
                              break

            if not found_code:
                 # Fallback: Search pattern in page content (less reliable)
                 try:
                     all_text = page.content()
                     code_match = self.patterns["product_code"].search(all_text)
                     if code_match:
                         product_code = code_match.group(1).strip()
                         found_code = True
                         self.logger.debug(f"Found product code: '{product_code}' using pattern search in page content.")
                     else:
                         self.logger.warning(f"Could not find product code using selectors or pattern search on {product_link}")
                 except Exception as text_ex:
                     self.logger.debug(f"Could not search for product code pattern in page content: {text_ex}")

            # 수량별 가격
            quantity_prices = {}
            qty_table_selector = self.selectors.get("quantity_table", {}).get("selector")
            if qty_table_selector:
                 qty_table_locator = page.locator(qty_table_selector).first
                 # Wait slightly longer for potentially dynamic tables
                 if qty_table_locator.is_visible(timeout=7000):
                     quantity_prices = self._extract_quantity_prices_from_locator(qty_table_locator)
                 else:
                     self.logger.debug(f"Quantity price table locator not visible: {qty_table_selector}")
            else:
                 self.logger.warning("'quantity_table' selector not defined.")

            # 상세 사양
            specifications = {}
            specs_table_selector = self.selectors.get("specs_table", {}).get("selector")
            if specs_table_selector:
                specs_table_locator = page.locator(specs_table_selector).first
                if specs_table_locator.is_visible(timeout=7000):
                    specifications = self._extract_specifications_from_locator(specs_table_locator)
                else:
                    self.logger.debug(f"Specifications table locator not visible: {specs_table_selector}")
            else:
                self.logger.warning("'specs_table' selector not defined.")

            # 이미지 갤러리
            image_gallery = []
            processed_urls = set()
            # Add list image as the first potential image if available
            list_image_url = item.get("image")
            if list_image_url:
                 processed_urls.add(list_image_url)
                 # No need to add to gallery yet, will be set as main_image_url later

            for selector in self.image_selectors:
                img_locators = page.locator(selector).all()
                for img_locator in img_locators:
                    # Ensure the locator is visible before extracting src
                    if img_locator.is_visible(timeout=500):
                        img_src = self._safe_get_attribute(img_locator, "src", timeout=1000) or self._safe_get_attribute(img_locator, "data-src", timeout=1000)
                        if img_src:
                            # Handle potential relative URLs correctly
                            try:
                                 img_url = urljoin(page.url, img_src.strip()) # Use page.url as base
                            except ValueError:
                                 self.logger.warning(f"Could not join base URL {page.url} with image src {img_src}")
                                 continue # Skip invalid URLs

                            if img_url not in processed_urls and not self._is_ui_image(img_url):
                                image_gallery.append(img_url)
                                processed_urls.add(img_url)

            # Set main image: prioritize detail page #main_img, then list image, then first gallery image
            main_image_url = None
            main_img_locator = page.locator("#main_img").first
            if main_img_locator.is_visible(timeout=1000):
                 main_img_src = self._safe_get_attribute(main_img_locator, "src")
                 if main_img_src:
                      try:
                           main_image_url = urljoin(page.url, main_img_src.strip())
                           self.logger.debug(f"Found main image URL from #main_img: {main_image_url}")
                      except ValueError:
                           self.logger.warning(f"Could not join base URL {page.url} with main image src {main_img_src}")

            if not main_image_url:
                 main_image_url = list_image_url # Use list image if detail main image failed
                 if main_image_url:
                      self.logger.debug(f"Using list image URL as main image: {main_image_url}")

            if not main_image_url and image_gallery:
                 main_image_url = image_gallery[0] # Use first gallery image as last resort
                 self.logger.debug(f"Using first gallery image as main image: {main_image_url}")

            # 상세 설명
            description = None
            desc_selector = self.selectors.get("description", {}).get("selector")
            if desc_selector:
                # Split potentially multiple selectors
                for sel in desc_selector.split(','):
                    sel = sel.strip()
                    if not sel: continue
                    desc_locator = page.locator(sel).first
                    if desc_locator.is_visible(timeout=5000):
                        # Prefer inner_html to preserve formatting, fallback to text
                        try:
                             description_html = desc_locator.inner_html(timeout=5000)
                             # Basic check to avoid empty divs
                             if description_html and len(description_html) > 50:
                                description = description_html
                                self.logger.debug(f"Found description using selector: {sel} (html)")
                                break
                        except (PlaywrightTimeoutError, PlaywrightError):
                             # Fallback to text content if inner_html fails or is empty
                             desc_text = self._safe_get_text(desc_locator)
                             if desc_text and len(desc_text) > 50:
                                  description = desc_text
                                  self.logger.debug(f"Found description using selector: {sel} (text)")
                                  break # Found description, stop checking selectors
                if not description:
                     self.logger.debug(f"Description content not found or too short using selectors: {desc_selector}")
            else:
                 self.logger.warning("'description' selector not defined.")

            # Determine final status
            if not main_image_url:
                 status = "Image Not Found"
                 self.logger.warning(f"Could not determine main image URL for product: {title} ({product_link})")
            else:
                 if price > 0:
                     status = "OK"  # 이미지와 가격이 모두 있으면 OK
                 else:
                     status = "Price Not Found"  # 이미지는 있지만 가격이 없는 경우

            # Product 객체 생성
            product = Product(
                id=product_id,
                name=title.strip() if title else "Unknown Product",
                price=price,
                source="koryo",
                original_input_data=item,
                url=product_link,
                image_url=main_image_url,
                image_gallery=image_gallery if image_gallery else None,
                product_code=product_code,
                quantity_prices=quantity_prices if quantity_prices else None,
                specifications=specifications if specifications else None,
                description=description,
                fetched_at=datetime.now().isoformat(),
                status=status, # 업데이트된 status 사용
            )
            self.logger.info(f"Successfully extracted details for product ID: {product.id} with status: {status}")
            return product

        except Exception as e:
            self.logger.error(f"Error extracting product details from page {product_link}: {e}", exc_info=self.config.debug)
            # Return None to indicate failure
            return None

    def _extract_quantity_prices_from_locator(self, table_locator: Locator) -> Dict[str, float]:
        """가격 테이블 Locator에서 수량별 가격 정보 추출 (Updated for .quantity_price__table)"""
        quantity_prices = {}
        try:
            # Assuming first row is header (quantity tiers)
            # Assuming second row is prices
            rows = table_locator.locator("tr").all()
            if len(rows) >= 2:
                qty_cells = rows[0].locator("td, th").all() # Header can use th or td
                price_cells = rows[1].locator("td, th").all()

                if len(qty_cells) == len(price_cells):
                    for i in range(len(qty_cells)):
                        qty_text = self._safe_get_text(qty_cells[i], timeout=500)
                        price_text = self._safe_get_text(price_cells[i], timeout=500)

                        if qty_text and price_text:
                            # Extract numbers robustly
                            qty_match = re.search(r"(\d[\d,]*)", qty_text.replace(",", ""))
                            qty = int(qty_match.group(1)) if qty_match else 0

                            price_match = self.patterns["price_number"].search(price_text)
                            price = float(price_match.group().replace(",", "")) if price_match else 0.0

                            if qty > 0 and price > 0:
                                quantity_prices[str(qty)] = price
                else:
                    self.logger.warning("Quantity table header and price row cell count mismatch.")
            else:
                 self.logger.warning("Quantity table has less than 2 rows.")

        except Exception as e:
            self.logger.error(f"Error extracting quantity prices from locator: {e}", exc_info=self.config.debug)
        return quantity_prices

    def _extract_specifications_from_locator(self, table_locator: Locator) -> Dict[str, str]:
        """사양 테이블 Locator에서 정보 추출 (Updated for .tbl_info)"""
        specs = {}
        try:
            rows = table_locator.locator("tr").all()
            for row_locator in rows:
                # Expecting structure: <tr><th>Key</th><td>Value</td></tr>
                header_locator = row_locator.locator("th").first
                value_locator = row_locator.locator("td").first

                # Check if both key and value cells exist
                if header_locator.count() > 0 and value_locator.count() > 0:
                     key = self._safe_get_text(header_locator, timeout=1000)
                     value = self._safe_get_text(value_locator, timeout=1000) # Consider inner_html if needed

                     if key and value:
                         # Clean up key (remove trailing ':')
                         cleaned_key = key.strip().rstrip(':').strip()
                         if cleaned_key:
                             specs[cleaned_key] = value.strip()
                     elif key and not value:
                          # Handle cases where value might be empty or complex (e.g., nested elements)
                          self.logger.debug(f"Specification key '{key.strip()}' found but value is empty or couldn't be extracted as simple text.")
                else:
                     # Handle potentially different row structures if necessary
                     self.logger.debug("Skipping row in specs table, expected th+td structure not found.")

        except Exception as e:
            self.logger.error(f"Error extracting specifications from locator: {e}", exc_info=self.config.debug)
        return specs

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
        """상품 ID로 상품 정보 가져오기 (Playwright)
           고려기프트는 ID로 직접 접근하는 URL 패턴이 불명확하여, 검색 기반으로 구현.
        """
        self.logger.info(f"Attempting to get product by ID (via direct URL and search): {product_id}")
        # 상품 ID가 URL의 no= 파라미터 값이라고 가정
        # 또는 상품 코드가 ID일 수도 있음. 여기서는 no= 값으로 가정.
        # 실제로는 상품 코드로 검색하는 것이 더 일반적일 수 있음.
        potential_url = f"{self.base_url}/ez/goods/goods_view.php?no={product_id}"

        # Use thread-local browser context for this operation
        self._close_thread_context()  # Ensure clean state
        browser, page = self._create_new_context()

        if not browser or not page:
            self.logger.error(f"Get product failed for ID {product_id}: Could not create browser context")
            return None

        product: Optional[Product] = None
        try:
            self.logger.info(f"Attempting direct URL access for product ID {product_id}: {potential_url}")
            page.goto(potential_url, wait_until='domcontentloaded')
            self._wait_for_load_state(page, 'networkidle')

            # 페이지 로드 후, 해당 ID의 상품이 맞는지 확인 (예: 상품명 또는 코드 확인)
            # 여기서는 간단히 페이지 내용을 기반으로 item 생성 후 상세 정보 추출 시도
            item_data = {
                "link": potential_url,
                "product_id": product_id,
                # 기본 정보는 알 수 없으므로 상세 추출에 의존
                "title": None,
                "price": None,
                "image": None
            }
            product = self._extract_product_details(page, item_data)
            if product:
                 self.logger.info(f"Successfully fetched and extracted product via direct URL for ID: {product_id}")
                 self.cache_sparse_data(f"koryo_detail|{product_id}", product)
                 # return product # Don't return yet, close context in finally block
            else:
                 self.logger.warning(f"Could not extract valid product details for ID: {product_id} via direct URL: {potential_url}. Will try searching.")
                 product = None # Ensure product is None if extraction failed

        except (PlaywrightTimeoutError, PlaywrightError) as e:
            self.logger.error(f"Playwright error accessing direct URL for product ID {product_id}: {e}. Will try searching.")
            product = None # Ensure product is None if direct access failed
        except Exception as e:
             self.logger.error(f"Unexpected error accessing direct URL for product ID {product_id}: {e}", exc_info=self.config.debug)
             product = None # Ensure product is None
        finally:
            # Clean up context from direct URL attempt
            self._close_thread_context()

        # If direct URL access/extraction failed, try searching using the ID as a query
        if product is None:
            self.logger.info(f"Attempting search using ID '{product_id}' as query.")
            # Search requires its own context, which it handles internally
            search_results = self.search_product(query=product_id, max_items=1)
            if search_results:
                 self.logger.info(f"Found product via search using ID as query: {product_id}")
                 product = search_results[0]
                 # Detail fetching (including status setting) happens within search_product -> _get_product_details_sync_playwright if needed
                 # Or, if search_product just returns list items, we might need to fetch details here.
                 # Let's assume search_product returns Product objects with details already fetched or cached.
                 # If it only returned list items, we'd need another call:
                 # product = self._get_product_details_sync_playwright(search_results[0].original_input_data)

            else:
                 self.logger.error(f"Product not found for ID {product_id} via direct URL or search.")
                 # Return None explicitly if not found
                 return None

        # If product was found either way, return it
        if product:
            self.logger.info(f"Final result for get_product(ID={product_id}): Found Product (Status: {product.status})")
            return product
        else:
            # This case should be covered by the search failure log above, but added for clarity
            self.logger.error(f"Final result for get_product(ID={product_id}): Product Not Found")
            return None

    def search_product(self, query: str, max_items: int = 50, keyword2: Optional[str] = None) -> List[Product]:
        """
        제품 검색 수행 및 결과 반환
        
        Args:
            query: 검색어
            max_items: 최대 결과 수
            keyword2: 추가 검색어 (선택 사항)
            
        Returns:
            List[Product]: 검색 결과 제품 목록
        """
        if not query:
            self.logger.warning("빈 검색어로 검색할 수 없습니다.")
            return []
            
        # 캐시 키 생성 (추가 검색어 포함)
        cache_key = f"search_{query}_{keyword2 or ''}"
        if self.cache:
            cached_results = self.cache.get(cache_key)
            if cached_results:
                self.logger.info(f"캐시에서 '{query}' 검색 결과 {len(cached_results)}개를 로드했습니다.")
                return cached_results
                
        processed_query = query.strip()
        if not processed_query:
            self.logger.warning(f"검색어가 비어있습니다: '{query}'")
            return []
            
        self.logger.info(f"Searching Koryo Gift for: {processed_query}")
        
        # 결과 변수 초기화
        all_results = []
        success = False
        
        # URL 인코딩
        encoded_query = urllib.parse.quote(processed_query)
        self.logger.info(f"Searching Koryo Gift for '{processed_query}' (Encoded: '{encoded_query}')")
        
        # 직접 접속할 대체 주소 - 사이트 홈페이지
        fallback_urls = [
            "https://www.koreagift.com",
            "https://koreagift.com",
            "https://www.adpanchok.co.kr", 
            "https://adpanchok.co.kr"
        ]
        
        # 재시도 횟수 설정
        retry_count = 0
        max_retries = 3
        
        # 각 기본 URL 시도
        for base_url in self.base_search_urls:
            # 최대 재시도 횟수 확인
            if retry_count >= max_retries:
                self.logger.warning(f"재시도 횟수 초과({max_retries}회): 검색 중단")
                break
                
            retry_count += 1
            
            try:
                # 브라우저 상태 체크 및 초기화
                browser, page = self._create_new_context()
                if not browser or not page:
                    self.logger.error("Failed to create browser context for search")
                    continue
                
                try:
                    # 접속 안정성을 위해 먼저 메인 사이트 접속 시도
                    try:
                        # base_url에서 도메인 부분만 추출
                        domain_parts = base_url.split('//')
                        if len(domain_parts) > 1:
                            domain = domain_parts[1].split('/')[0]
                            main_url = f"https://{domain}"
                            self.logger.info(f"먼저 메인 사이트 접속 시도: {main_url}")
                            
                            # 메인 사이트 접속
                            page.goto(
                                main_url,
                                wait_until="domcontentloaded",
                                timeout=20000  # 20초 타임아웃
                            )
                            # 짧은 대기 시간
                            time.sleep(1)
                    except Exception as main_error:
                        self.logger.warning(f"메인 사이트 접속 실패: {main_error}")
                        # 오류가 발생해도 계속 진행
                    
                    # 검색 URL 구성
                    search_url = f"{base_url}?search_str={encoded_query}"
                    if not base_url.endswith((".php", ".html", ".asp", ".aspx")):
                        # 루트 도메인인 경우 다른 URL 형식 사용
                        search_url = f"{base_url}/ez/goods/goods_search.php?search_str={encoded_query}"
                    
                    # 검색 로깅
                    self.logger.info(f"Trying search URL: {search_url}")
                    
                    # 검색 페이지 로드 시도 (여러 번 재시도)
                    goto_success = False
                    goto_retries = 3
                    
                    for goto_attempt in range(goto_retries):
                        try:
                            response = page.goto(
                                search_url, 
                                wait_until="domcontentloaded", 
                                timeout=20000  # 20초로 감소
                            )
                            goto_success = True
                            break
                        except Exception as goto_error:
                            self.logger.warning(f"Navigation error for {search_url} (attempt {goto_attempt+1}/{goto_retries}): {goto_error}")
                            # 짧은 대기 후 재시도
                            time.sleep(1)
                    
                    if not goto_success:
                        self.logger.warning(f"Failed to navigate to {search_url} after {goto_retries} attempts")
                        continue
                    
                    # 응답 확인
                    if not response:
                        self.logger.warning(f"No response for search URL: {search_url}")
                        continue
                        
                    if not response.ok:
                        self.logger.warning(f"HTTP error for search URL: {search_url} (Status: {response.status})")
                        continue
                    
                    # 페이지 로드 대기 (더 짧은 타임아웃)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)  # 10초로 감소
                    except Exception as e:
                        self.logger.info(f"Network idle timeout for {search_url}: {e}")
                        # 계속 진행 (완전히 로드되지 않아도 결과가 있을 수 있음)
                    
                    # 짧은 대기 시간 추가 (페이지 안정화)
                    time.sleep(0.5)
                    
                    # 검색 결과 추출
                    try:
                        # 검색 결과 컨테이너 확인 (다양한 선택자 시도)
                        result_selectors = [
                            "ul.prd_list",
                            "div.prd_list_wrap",
                            "table.mall_list",
                            "div.product_list",
                            ".prd-list",
                            ".best100_tab",
                            ".goods-list", 
                            ".product-grid"
                        ]
                        
                        found_container = False
                        result_container = None
                        
                        for selector in result_selectors:
                            try:
                                result_container = page.locator(selector)
                                if result_container.count() > 0:
                                    self.logger.info(f"Found container with selector: {selector}")
                                    found_container = True
                                    break
                            except Exception as container_error:
                                self.logger.debug(f"Error checking container {selector}: {container_error}")
                        
                        # 컨테이너를 찾지 못하면 body 전체에서 item 탐색 시도
                        if not found_container:
                            self.logger.info(f"No specific container found, searching in entire body")
                            result_container = page.locator("body")  # 전체 페이지에서 탐색
                        
                        # 검색 결과 아이템 확인 (다양한 선택자 시도)
                        item_selectors = [
                            "li.prd",
                            "ul.prd_list li", 
                            "table.mall_list td.prd", 
                            "div.product_list .item", 
                            ".prd-list .prd-item", 
                            ".product",
                            ".goods-item",
                            ".product-grid-item"
                        ]
                        
                        result_items = None
                        count = 0
                        
                        for selector in item_selectors:
                            try:
                                # 컨테이너 내에서 검색
                                if found_container:
                                    result_items = result_container.locator(selector)
                                # 전체 페이지에서 검색
                                else:
                                    result_items = page.locator(selector)
                                
                                count = result_items.count()
                                if count > 0:
                                    self.logger.info(f"Found {count} items with selector: {selector}")
                                    break
                            except Exception as item_selector_error:
                                self.logger.debug(f"Error with item selector {selector}: {item_selector_error}")
                        
                        # 아이템을 찾지 못하면 마지막 대안으로 페이지 내 모든 a 태그 중 상품 이미지가 있는 것 찾기
                        if not result_items or count == 0:
                            try:
                                self.logger.info("Trying to find product links with images as last resort")
                                # 이미지가 포함된 링크 찾기
                                image_links = page.locator("a:has(img)")
                                count = image_links.count()
                                if count > 0:
                                    self.logger.info(f"Found {count} potential product links with images")
                                    result_items = image_links
                                else:
                                    self.logger.info(f"No search results for '{processed_query}' on {base_url}")
                                    continue
                            except Exception as fallback_error:
                                self.logger.warning(f"Error in fallback search: {fallback_error}")
                                self.logger.info(f"No search results for '{processed_query}' on {base_url}")
                                continue
                        
                        self.logger.info(f"Found {count} search results for '{processed_query}' on {base_url}")
                        
                        # 각 결과 항목 처리
                        products = []
                        for i in range(min(count, max_items)):
                            try:
                                item_locator = result_items.nth(i)
                                
                                # 직접 데이터 추출 시도
                                try:
                                    item_data = self._extract_list_item(item_locator)
                                except Exception as extract_error:
                                    self.logger.warning(f"Error extracting item #{i}: {extract_error}")
                                    # 간단한 정보만 추출하는 대체 로직
                                    item_data = self._extract_minimal_item_data(item_locator, i)
                                
                                if item_data:
                                    # 제품 상세 정보 얻기
                                    try:
                                        product = self._get_product_details_sync_playwright(item_data)
                                        if product and self._validate_and_normalize_product(product):
                                            products.append(product)
                                            self.logger.debug(f"Added product: {product.name}")
                                    except Exception as product_error:
                                        self.logger.warning(f"Error getting product details for item #{i}: {product_error}")
                                        # 최소한의 제품 정보로 생성
                                        if 'title' in item_data and item_data['title']:
                                            minimal_product = Product(
                                                id=item_data.get('product_id', f"item_{i}_{int(time.time())}"),
                                                name=item_data['title'],
                                                source="koryo",
                                                price=item_data.get('price', 0),
                                                url=item_data.get('link', ''),
                                                image_url=item_data.get('image', '')
                                            )
                                            products.append(minimal_product)
                                            self.logger.debug(f"Added minimal product: {minimal_product.name}")
                            except Exception as item_ex:
                                self.logger.warning(f"Error processing search result item #{i}: {item_ex}")
                                continue
                                
                        if products:
                            all_results.extend(products)
                            success = True
                            self.logger.info(f"Successfully extracted {len(products)} products from {base_url}")
                            
                            # 충분한 결과를 얻었으면 중단
                            if len(all_results) >= max_items:
                                self.logger.info(f"Reached maximum items limit ({max_items})")
                                break
                                
                    except Exception as extract_ex:
                        self.logger.error(f"Error extracting search results from {base_url}: {extract_ex}")
                        continue
                        
                finally:
                    # 페이지 및 컨텍스트 닫기
                    try:
                        if page:
                            page.close()
                        self._close_thread_context()
                    except Exception as e:
                        self.logger.warning(f"Error closing page/context: {e}")
                
                # 성공했고 충분한 결과가 있으면 다음 URL 시도하지 않음
                if success and len(all_results) >= max_items:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error searching {base_url} for '{processed_query}': {e}", exc_info=True)
                continue

        # 중복 결과 제거 (URL 기준)
        unique_products = []
        seen_urls = set()
        
        for product in all_results:
            if product.url not in seen_urls:
                seen_urls.add(product.url)
                unique_products.append(product)

        # 결과 정렬 (가격 오름차순)
        unique_products.sort(key=lambda p: p.price if p.price is not None else float('inf'))
        
        # 결과 캐싱
        if self.cache and unique_products:
            self.logger.info(f"Caching {len(unique_products)} search results for '{query}'")
            self.cache.set(cache_key, unique_products, ttl=3600)  # 1시간 캐싱
        
        # 검색 결과가 없으면 "No match" 제품 반환
        if not unique_products:
            self.logger.warning(f"No products found for '{query}'")
            no_match = self._create_no_match_product(query)
            return [no_match]
            
        return unique_products[:max_items]

    def _create_no_match_product(self, query: str, is_error: bool = False) -> Product:
        """검색 결과 없거나 오류 발생 시 기본 제품 생성"""
        prefix = "검색 오류" if is_error else "동일상품 없음"
        
        # 현재 타임스탬프로 고유한 ID 생성
        timestamp = int(datetime.now().timestamp())
        product_id = f"no_match_{timestamp}"
        
        # 고유 ID 기반 가상 이미지 URL 생성
        image_url = f"https://koreagift.com/ez/upload/mall/shop_{timestamp}_0.jpg"
        
        return Product(
            id=product_id,
            name=f"{prefix} - {query}",
            source="koryo",
            price=0,
            url="",
            image_url=image_url  # 타임스탬프 기반 이미지 URL 사용
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
        if not product.image_url or product.image_url == "https://koreagift.com/img/ico_shop.gif":
            # 기본 이미지를 사용하지 않고 상품 ID를 활용한 실제 이미지 URL 패턴 적용
            if hasattr(product, 'id') and product.id:
                # 상품 ID가 있는 경우 해당 ID를 활용한 예상 이미지 URL 생성
                product_id = product.id
                if "adpanchok.co.kr" in product.url:
                    # 예: https://adpanchok.co.kr/ez/upload/mall/shop_상품ID_0.jpg
                    product.image_url = f"https://adpanchok.co.kr/ez/upload/mall/shop_{product_id}_0.jpg"
                else:
                    # 예: https://koreagift.com/ez/upload/mall/shop_상품ID_0.jpg
                    product.image_url = f"https://koreagift.com/ez/upload/mall/shop_{product_id}_0.jpg"
                self.logger.info(f"상품 '{product.name}'의 이미지 URL 없음, ID 기반 URL 생성: {product.image_url}")
            else:
                # ID를 사용할 수 없는 경우 기본 이미지 사용
                product.image_url = "https://koreagift.com/img/ico_shop.gif"  # 기본 이미지 설정
                self.logger.warning(f"상품 '{product.name}'의 이미지 URL이 없어 기본 이미지를 사용합니다.")
        else:
            # 2.1 상대 경로를 절대 경로로 변환
            if not product.image_url.startswith(('http://', 'https://', '//')):
                if product.image_url.startswith('/'):
                    # 루트 상대 경로
                    if "adpanchok.co.kr" in product.url:
                        product.image_url = f"https://adpanchok.co.kr{product.image_url}"
                    else:
                        product.image_url = f"https://koreagift.com{product.image_url}"
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
                            img_url = f"https://koreagift.com{img_url}"
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

    def _get_product_details_sync_playwright(self, item: Dict) -> Optional[Product]:
        """상품 상세 정보 추출 (Playwright 동기)"""
        # ... (rest of the _get_product_details_sync_playwright method remains the same) ...
        # Ensure this method also uses its own context or reuses carefully
        link = item.get("link")
        if not link:
            self.logger.warning("Item dictionary is missing 'link'. Cannot fetch details.")
            return None

        # Cache check
        cache_key = f"koryo_details_{link}"
        if self.cache and (cached_product := self.cache.get(cache_key)):
             if isinstance(cached_product, Product):
                 self.logger.info(f"Cache hit for Koryo details: {link}")
                 # Ensure status is updated if needed, maybe re-validate cache freshness?
                 # cached_product.status = ProductStatus.FETCHED # Or keep cached status
                 return cached_product
             else:
                 self.logger.warning(f"Invalid data type found in cache for key {cache_key}. Fetching fresh data.")


        self._close_thread_context() # Ensure clean state before getting details
        browser, page = self._create_new_context()
        if not browser or not page:
            self.logger.error(f"Failed to get details for {link}: Could not create browser context")
            return None

        product: Optional[Product] = None
        try:
            self.logger.info(f"Fetching details for Koryo product: {link}")
            response = page.goto(link, wait_until='domcontentloaded', timeout=self.config.navigation_timeout)

            if not response or not response.ok:
                 status = response.status if response else "N/A"
                 self.logger.error(f"Failed to load product page {link}. Status: {status}")
                 product = Product(id=item.get('id', f"error_{link}"), name=item.get('title', 'Failed to Load'), url=link, source='koryo', status=ProductStatus.FETCH_ERROR) # Assuming ProductStatus enum exists
                 return product # Return error product


            # Wait for essential content if necessary (example: price or title)
            # self._wait_for_selector(page, self.selectors['details']['title'], timeout=self.config.wait_timeout)

            # Extract details using the existing logic
            product = self._extract_product_details(page, item)

            if product:
                 # 상태가 이미 설정되어 있지 않거나 기본 메시지인 경우에만 업데이트
                 if not product.status or product.status == "Extraction Failed":
                     product.status = ProductStatus.FETCHED  # Mark as successfully fetched
                 
                 self.logger.info(f"Successfully extracted details for: {product.name} with status: {product.status}")
                 if self.cache:
                     self.cache.set(cache_key, product)  # Cache the successful result
            else:
                 # Extraction failed after successful page load
                 self.logger.error(f"Failed to extract details from page: {link}")
                 product = Product(id=item.get('id', f"extract_error_{link}"), name=item.get('title', 'Extraction Failed'), url=link, source='koryo', status=ProductStatus.EXTRACT_ERROR)
                 # Don't cache extraction errors to allow retry
                 return product


        except Exception as e:
            self.logger.error(f"Error fetching/extracting Koryo details for {link}: {e}", exc_info=True)
            product = Product(id=item.get('id', f"exception_{link}"), name=item.get('title', 'Exception'), url=link, source='koryo', status=ProductStatus.FETCH_ERROR)
            return product
        finally:
            # Close the context created for this detail fetch
            if page: page.close()
            # Assuming context closing is handled if _create_new_context stores it
            # Or manually close: if context: context.close()
            pass # Ensure cleanup

        return product

    def _extract_minimal_item_data(self, item_locator: Locator, index: int) -> Optional[Dict]:
        """최소한의 아이템 데이터 추출 시도 (오류 상황에서 사용)"""
        minimal_data = {}
        try:
            # 제목 추출 시도
            try:
                # 1. 직접 텍스트 추출
                title = item_locator.text_content()
                
                # 2. 내부 텍스트 노드 중 가장 긴 것 찾기
                if not title or len(title.strip()) < 3:
                    text_nodes = item_locator.locator("text")
                    max_len = 0
                    for i in range(text_nodes.count()):
                        text = text_nodes.nth(i).text_content()
                        if text and len(text.strip()) > max_len:
                            title = text.strip()
                            max_len = len(title)
                
                # 3. 내부 링크 텍스트 추출
                if not title or len(title.strip()) < 3:
                    a_tags = item_locator.locator("a")
                    for i in range(a_tags.count()):
                        a_text = a_tags.nth(i).text_content()
                        if a_text and len(a_text.strip()) > 3:
                            title = a_text.strip()
                            break
                
                minimal_data["title"] = title.strip() if title else f"상품 {index+1}"
            except:
                minimal_data["title"] = f"상품 {index+1}"
            
            # 링크 추출 시도
            try:
                # 1. 아이템 자체가 링크인 경우
                href = item_locator.get_attribute("href")
                
                # 2. 내부 첫 번째 링크 사용
                if not href:
                    a_tag = item_locator.locator("a").first
                    href = a_tag.get_attribute("href")
                
                if href:
                    minimal_data["link"] = urljoin(self.base_url, href.strip())
                    minimal_data["product_id"] = hashlib.md5(minimal_data["link"].encode()).hexdigest()
            except:
                # 링크를 찾지 못하면 임시 ID 생성
                minimal_data["link"] = ""
                minimal_data["product_id"] = f"temp_id_{index}_{int(time.time())}"
            
            # 이미지 추출 시도
            try:
                img_tag = item_locator.locator("img").first
                if img_tag:
                    img_src = img_tag.get_attribute("src")
                    if img_src:
                        minimal_data["image"] = urljoin(self.base_url, img_src.strip())
            except:
                minimal_data["image"] = ""
            
            # 가격 추출 시도
            try:
                # 1. 가격 관련 클래스 확인
                price_text = None
                price_selectors = [".price", "span.price", "div.price", "p.price", "strong.price"]
                
                for selector in price_selectors:
                    try:
                        price_elem = item_locator.locator(selector).first
                        if price_elem:
                            price_text = price_elem.text_content()
                            if price_text:
                                break
                    except:
                        continue
                
                # 2. 가격 텍스트에서 숫자 추출
                if price_text:
                    price_match = re.search(r"[\d,]+", price_text)
                    if price_match:
                        price_str = price_match.group().replace(",", "")
                        minimal_data["price"] = float(price_str)
                    else:
                        minimal_data["price"] = 0
                else:
                    minimal_data["price"] = 0
            except:
                minimal_data["price"] = 0
            
            return minimal_data
        except Exception as e:
            self.logger.warning(f"Error extracting minimal item data: {e}")
            # 절대 최소한의 데이터
            return {
                "title": f"상품 {index+1}",
                "link": "",
                "product_id": f"temp_id_{index}_{int(time.time())}",
                "image": "",
                "price": 0
            }

        