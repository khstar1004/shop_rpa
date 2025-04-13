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
            ".name" # main page best100 analysis
        ]
        self.price_selectors = [
            "#main_price", # product page analysis
            "dl.detail_price dd",
            ".price_num",
            ".product-price",
            ".price", # main page best100 analysis
            ".product-price-num"
        ]
        self.code_selectors = [
            '//div[contains(text(), "상품코드")]', # product page analysis (XPath needed for text containment)
            # "div.product_code", # Original, less specific
            ".product-code",
            ".item-code",
            "span.code",
            'td:contains("상품코드") + td',
            'li:contains("상품코드")'
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
            "div.prd_detail img"
        ]
        self.selectors = {
            "product_list": {
                # Priority to more specific structures if known context (e.g., best100)
                # General selectors first for broader compatibility
                "selector": "div.prd_list_wrap li.prd, ul.prd_list li, table.mall_list td.prd, div.product_list .item, .prd-list .prd-item, .product, .best100_tab .product", # crowling_kogift.py .product 추가, main page best100 analysis 추가
            },
            "product_title_list": { # 목록용 제목
                 "selector": ".name, p.name a, div.name a, td.name a, .prd_name a, a.product-title, div.name > a", # main page best100 analysis 추가, crowling_kogift.py div.name > a 추가
            },
            "product_link_list": { # 목록용 링크
                 "selector": "a, p.name a, div.name a, td.name a, .prd_name a, a.product-title, div.prd_list_wrap li.prd > a, div.pic > a", # Ensure 'a' itself is included for cases like '.product a', crowling_kogift.py div.pic > a 추가
                 "attribute": "href",
            },
            "price_list": { # 목록용 가격
                "selector": ".price, p.price, div.price, td.price, .prd_price, span.price", # main page best100 analysis 추가, crowling_kogift.py div.price 추가
            },
            "thumbnail_list": { # 목록용 썸네일
                "selector": ".img img, .pic img, img.prd_img, td.img img, .thumb img, img.product-image, div.pic > a > img", # main page best100 analysis 추가, crowling_kogift.py div.pic > a > img 추가
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
            "https://koreagift.com/ez/goods/goods_search.php",
            "https://adpanchok.co.kr/ez/goods/goods_search.php"
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
            self._browser = self._playwright.chromium.launch(headless=self.config.headless)
            context = self._browser.new_context(
                user_agent=self.config.user_agent,
                viewport={'width': 1920, 'height': 1080} # 필요시 뷰포트 설정
            )
            # 기본 타임아웃 설정
            context.set_default_timeout(self.config.timeout)
            context.set_default_navigation_timeout(self.config.navigation_timeout)

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
            browser = playwright.chromium.launch(headless=self.config.headless)
            context = browser.new_context(
                user_agent=self.config.user_agent,
                viewport={'width': 1920, 'height': 1080}
            )
            
            # Set timeouts
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
        """상품 검색 (Playwright 사용, 다중 URL 시도)"""
        self._close_thread_context() # Ensure clean state
        browser, page = self._create_new_context()

        if not browser or not page:
            self.logger.error(f"Search failed for '{query}': Could not create browser context")
            return []

        self.logger.info(f"Searching Koryo sites for: '{query}' (Secondary: '{keyword2}')")
        products = []
        navigation_successful = False
        active_search_url = None # Store the URL that succeeded

        try:
            # --- Build Search Params --- #
            search_params = {
                "search_keyword": query,
                "search_type": "all"
            }
            # --- End Build Search Params --- #

            # === Navigation Loop: Try Base URLs === #
            for base_url in self.base_search_urls:
                current_search_url = f"{base_url}?{urllib.parse.urlencode(search_params)}"
                self.logger.info(f"Attempting navigation to search URL: {current_search_url}")
                page_response = None # Reset response for this attempt

                try:
                    nav_timeout = self.config.navigation_timeout
                    page_response = page.goto(current_search_url, wait_until='networkidle', timeout=nav_timeout)

                    # Check 1: Response OK?
                    if page_response is None or not page_response.ok:
                        status_code = page_response.status if page_response else "N/A"
                        self.logger.warning(f"Navigation to {current_search_url} failed or returned non-OK status: {status_code}. Trying next URL.")
                        continue # Try next base URL

                    # Check 2: Content indicates "Not Found"?
                    page_content = page.content().lower()
                    page_title = page.title().lower()
                    if "not found" in page_content or "not found" in page_title:
                        self.logger.warning(f"Search page content indicates 'Not Found' for URL: {current_search_url} (Query: '{query}'). Title: '{page.title()}'. Trying next URL.")
                        continue # Try next base URL

                    # Success for this URL
                    self.logger.info(f"Successfully navigated to and verified search page: {current_search_url}")
                    navigation_successful = True
                    active_search_url = current_search_url # Store the successful URL
                    break # Exit the loop, proceed with scraping

                # --- Handle Navigation Exceptions --- #
                except Exception as e:
                    error_str = str(e)
                    if (
                        "net::ERR_CONNECTION_RESET" in error_str or
                        "net::ERR_EMPTY_RESPONSE" in error_str or
                        isinstance(e, PlaywrightTimeoutError)
                    ):
                        self.logger.warning(f"Network/Timeout error navigating to {current_search_url}: {type(e).__name__}. Trying next URL.")
                        continue # Try next URL
                    elif "net::ERR_ABORTED" in error_str:
                        self.logger.warning(f"Navigation aborted ('networkidle') for {current_search_url}. Retrying ('domcontentloaded').")
                        try:
                            page_response = page.goto(current_search_url, wait_until='domcontentloaded', timeout=nav_timeout)
                            # Check response and content again after retry
                            if page_response is None or not page_response.ok:
                                status_code = page_response.status if page_response else "N/A"
                                self.logger.warning(f"Retry navigation failed for {current_search_url}. Status: {status_code}. Trying next URL.")
                                continue
                            page_content = page.content().lower()
                            page_title = page.title().lower()
                            if "not found" in page_content or "not found" in page_title:
                                self.logger.warning(f"Search page 'Not Found' after retry for {current_search_url}. Trying next URL.")
                                continue
                            # Retry successful
                            self.logger.info(f"Successfully navigated (on retry) to: {current_search_url}")
                            navigation_successful = True
                            active_search_url = current_search_url
                            break # Exit loop
                        except Exception as inner_e:
                            self.logger.error(f"Retry navigation failed definitively for {current_search_url}: {inner_e}. Trying next URL.")
                            continue # Try next URL
                    else:
                        # Unexpected error during navigation attempt
                        self.logger.error(f"Unexpected navigation error for {current_search_url}: {e}. Trying next URL.")
                        continue # Try next URL
                # --- End Handle Navigation Exceptions --- #
            # === End Navigation Loop === #

            # --- Check if Navigation Succeeded Overall --- #
            if not navigation_successful:
                self.logger.error(f"Failed to navigate successfully to ANY search URL for query: '{query}'.")
                # No need to proceed further, cleanup will happen in finally
                return [] # Return empty list immediately
            # --- End Navigation Check --- #

            # \"\"\" --- Start Scraping Logic (Only runs if navigation_successful is True) --- \"\"\"
            self.logger.info(f"Starting scraping process on successfully loaded page: {active_search_url}")

            # --- Step 2: Check product count (for potential re-search) --- #
            product_count = 0
            try:
                count_locator = page.locator(self.selectors["product_count"]).first
                count_text = self._safe_get_text(count_locator, timeout=10000)
                if count_text:
                    count_match = re.search(r'\\d+', count_text.replace(',', ''))
                    if count_match:
                        product_count = int(count_match.group(0))
                        self.logger.info(f"Found {product_count} products after initial search.")
                    else:
                        self.logger.warning(f"Could not parse product count from text: '{count_text}'")
                else:
                    self.logger.warning(f"Could not get text for product count locator.")
            except Exception as e:
                self.logger.warning(f"Could not determine product count after initial search: {e}")
            # --- End Step 2 --- #

            # --- Step 3: Perform re-search if necessary --- #
            if keyword2 and keyword2.strip() and product_count >= 100:
                self.logger.info(f"Product count ({product_count}) >= 100. Performing re-search with: '{keyword2}'")
                re_search_input_loc = self.selectors["re_search_input"]
                re_search_button_loc = self.selectors["re_search_button"]
                re_search_input = self._wait_for_selector(page, re_search_input_loc)
                re_search_button = page.locator(re_search_button_loc).first
                if re_search_input and re_search_button.is_visible(timeout=5000):
                    re_search_input.fill(keyword2)
                    if self._click_locator(re_search_button):
                        self._wait_for_load_state(page, 'networkidle')
                        self.logger.info(f"Re-search submitted for '{keyword2}'.")
                        # Re-check product count (optional)
                        try:
                            count_locator = page.locator(self.selectors["product_count"]).first
                            count_text = self._safe_get_text(count_locator, timeout=10000)
                            if count_text:
                                count_match = re.search(r'\\d+', count_text.replace(',', ''))
                                if count_match:
                                    product_count = int(count_match.group(0))
                                    self.logger.info(f"Found {product_count} products after re-search.")
                                else:
                                     self.logger.warning(f"Could not parse count after re-search: '{count_text}'")
                            else:
                                 self.logger.warning(f"Could not get count text after re-search.")
                        except Exception as e:
                            self.logger.warning(f"Could not determine count after re-search: {e}")
                    else:
                        self.logger.warning("Failed to click re-search button.")
                else:
                    self.logger.warning("Re-search elements not found or visible.")
            # --- End Step 3 --- #

            # --- Step 4: Paginate and extract product list items --- #
            page_num = 1
            processed_links = set()
            while True:
                if max_items > 0 and len(products) >= max_items:
                    self.logger.info(f"Reached max items ({max_items}). Stopping pagination.")
                    break

                self.logger.info(f"Scraping page {page_num} for search '{query}' on {page.url}")
                list_selector = self.selectors["product_list"]["selector"]
                try:
                    page.wait_for_selector(list_selector, state='visible', timeout=15000)
                except PlaywrightTimeoutError:
                    self.logger.warning(f"Product list ({list_selector}) did not become visible on page {page_num}. Assuming end of results.")
                    break

                item_locators = page.locator(list_selector).all()
                if not item_locators:
                    if page_num == 1:
                        self.logger.info("No product items found on the first page.")
                    else:
                        self.logger.info(f"No more product items found on page {page_num}.")
                    break

                found_on_page = 0
                items_to_fetch_details = []
                for item_locator in item_locators:
                    if max_items > 0 and len(products) + len(items_to_fetch_details) >= max_items:
                        break
                    try:
                        product_data = self._extract_list_item(item_locator)
                        if product_data and product_data.get("link") and product_data["link"] not in processed_links:
                            items_to_fetch_details.append(product_data)
                            processed_links.add(product_data["link"])
                            found_on_page += 1
                    except Exception as e:
                        self.logger.warning(f"Error extracting list item on page {page_num}: {e}", exc_info=self.config.debug)

                self.logger.info(f"Found {found_on_page} new items on page {page_num}. Fetching details...")

                # Fetch details (Consider parallelization for performance)
                for item_data in items_to_fetch_details:
                    if max_items > 0 and len(products) >= max_items:
                        break
                    detailed_product = self._get_product_details_sync_playwright(item_data)
                    if detailed_product:
                        products.append(detailed_product)
                    else:
                        self.logger.warning(f"Failed get details for: {item_data.get('link')}")

                if found_on_page == 0 and page_num > 1:
                    self.logger.info(f"No new links found on page {page_num}. Ending pagination.")
                    break

                # Next Page Logic
                js_next_selector = self.selectors["next_page_js"]["selector"]
                href_next_selector = self.selectors["next_page_href"]["selector"]
                js_next_page_locator = page.locator(js_next_selector).last
                href_next_page_locator = page.locator(href_next_selector).last
                next_page_clicked = False
                # Try JS paging
                if js_next_page_locator.is_visible(timeout=2000):
                    js_code_template = self.selectors["next_page_js"].get("js_code", "")
                    if js_code_template:
                        js_code = js_code_template.replace("{page_num + 1}", str(page_num + 1))
                        try:
                            page.evaluate(js_code)
                            self._wait_for_load_state(page)
                            next_page_clicked = True
                            self.logger.info(f"Moved to page {page_num + 1} (JS).")
                        except Exception as js_e:
                            self.logger.warning(f"JS paging failed: {js_e}. Trying href.")
                    else:
                        self.logger.warning("JS code template missing.")
                # Try href paging
                if not next_page_clicked and href_next_page_locator.is_visible(timeout=2000):
                    if self._click_locator(href_next_page_locator):
                        self._wait_for_load_state(page)
                        next_page_clicked = True
                        self.logger.info(f"Moved to page {page_num + 1} (href).")
                    else:
                        self.logger.warning(f"Failed to click href next link.")

                if not next_page_clicked:
                    self.logger.info(f"No more pages found after page {page_num}.")
                    break
                page_num += 1
            # --- End Pagination Loop --- #
            # \"\"\" --- End Scraping Logic --- \"\"\"

        # --- General Exception Handling --- #
        except Exception as e:
            self.logger.error(f"Major error during Koryo search process for '{query}': {e}", exc_info=True)
            # products list will contain whatever was scraped before the error
        # --- End General Exception Handling --- #

        # --- Cleanup --- #
        finally:
            self._close_thread_context() # Ensure context is always closed
        # --- End Cleanup --- #

        self.logger.info(f"Finished Koryo search for '{query}'. Found {len(products)} products.")
        return products

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

        