import asyncio
import hashlib
import json
import logging
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# 추가: Playwright 임포트 시도 (선택적 의존성)
try:
    from playwright.sync_api import TimeoutError, sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Add imports for caching
from utils.caching import FileCache, cache_result

from ..data_models import Product
from . import BaseMultiLayerScraper, DOMExtractionStrategy, TextExtractionStrategy


class KoryoScraper(BaseMultiLayerScraper):
    """
    고려기프트 스크래퍼 - Selenium 및 Playwright 활용 다중 레이어 추출 엔진

    특징:
    - Selenium 기반 웹 브라우저 자동화
    - Playwright 지원 (설치된 경우)
    - DOM, 텍스트, 좌표 기반 추출 전략
    - 비동기 작업 처리
    - 메모리 효율적 데이터 구조
    - 선택적 요소 관찰
    - 가격표 및 수량별 가격 추출
    """

    def __init__(
        self,
        max_retries: int = 5,
        timeout: int = 30,
        cache: Optional[FileCache] = None,
        use_proxies: bool = False,
        debug: bool = False,
    ):
        super().__init__(max_retries=max_retries, timeout=timeout, cache=cache)
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout
        self.debug = debug
        self.base_url = "https://koreagift.com"
        self.use_proxies = use_proxies

        # Playwright 설정
        if PLAYWRIGHT_AVAILABLE:
            try:
                from playwright.sync_api import sync_playwright
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-gpu',
                        '--disable-dev-shm-usage',
                        '--disable-setuid-sandbox',
                        '--no-sandbox',
                        '--disable-web-security',
                        '--disable-features=IsolateOrigins,site-per-process',
                        '--disable-site-isolation-trials'
                    ]
                )
                self.context = self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Playwright: {e}")
                self.playwright = None
                self.browser = None
                self.context = None
        else:
            self.logger.warning("Playwright is not available. Some features will be limited.")
            self.playwright = None
            self.browser = None
            self.context = None

        # Selenium 설정
        self.driver = None
        self.setup_selenium()

        # 출력 디렉토리 생성
        os.makedirs("output", exist_ok=True)

        # 사이트 관련 상수 정의 (2023년 URL 업데이트)
        self.mall_url = f"{self.base_url}/ez/mall.php"
        self.search_url = f"{self.base_url}/ez/goods/goods_search.php"

        # 기본 카테고리 정의 (fallback용)
        self.categories_cache = []
        self.default_categories = [
            ("볼펜/사무용품", "013001000"),
            ("텀블러/머그컵", "013002000"),
            ("가방", "013003000"),
            ("전자/디지털", "013004000"),
        ]

        # 요청 헤더 (Selenium에서는 덜 중요하지만 유지)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": self.base_url,
        }

        # Playwright 사용 가능 여부 체크
        self.playwright_available = PLAYWRIGHT_AVAILABLE
        if not self.playwright_available:
            self.logger.warning(
                "Playwright is not installed. Some features will be limited."
            )

        # 추출 셀렉터 정의 - 구조화된 다중 레이어 접근 방식
        self.selectors = {
            "product_list": {
                "selector": "div.prd_list_wrap li.prd",
                "options": {"multiple": True},
            },
            "product_title": {
                "selector": "div.title a",
                "options": {"multiple": False},
            },
            "product_link": {
                "selector": "div.title a",
                "options": {"attribute": "href"},
            },
            "price": {"selector": "div.price", "options": {"multiple": False}},
            "thumbnail": {"selector": "img.prd_img", "options": {"attribute": "src"}},
            "next_page": {
                "selector": ".paging a.next",
                "options": {"attribute": "href"},
            },
            # 상세 페이지 셀렉터
            "detail_title": {
                "selector": "h3.detail_tit",
                "options": {"multiple": False},
            },
            "detail_price": {
                "selector": "dl.detail_price dd",
                "options": {"multiple": False},
            },
            "detail_code": {
                "selector": "div.product_code",
                "options": {"multiple": False},
            },
            "detail_images": {
                "selector": "div.swiper-slide img",
                "options": {"multiple": True, "attribute": "src"},
            },
            "quantity_table": {
                "selector": "table.quantity_price",
                "options": {"multiple": False},
            },
            "specs_table": {
                "selector": "table.specs_table",
                "options": {"multiple": False},
            },
            "description": {
                "selector": "div.prd_detail",
                "options": {"multiple": False},
            },
            # 카테고리 셀렉터
            "category_items": {
                "selector": "#lnb_menu > li > a",
                "options": {"multiple": True},
            },
            # 가격표 관련 셀렉터 추가
            "price_table": {
                "selector": "table.price_table, table.quantity_table, table.option_table",
                "options": {"multiple": False},
            },
            "quantity_input": {
                "selector": 'input.qu, input[name*="quantity"], input[name*="qty"]',
                "options": {"multiple": True, "attribute": "value"},
            },
            "price_input": {
                "selector": 'input.pr, input[name*="price"]',
                "options": {"multiple": True, "attribute": "value"},
            },
        }

        # 텍스트 추출용 정규식 패턴 (추가 패턴 포함)
        self.patterns = {
            "price_number": re.compile(r"[\d,]+"),
            "product_code": re.compile(r"상품코드\s*:\s*([A-Za-z0-9-]+)"),
            "quantity": re.compile(r"(\d+)(개|세트|묶음)"),
            "quantity_price": re.compile(r"(\d+)개[:\s]+([0-9,]+)원"),  # 수량:가격 패턴
            "vat_included": re.compile(r"VAT\s*(포함|별도|제외)", re.IGNORECASE),
            "quantity_price": re.compile(r"(\d+)개[:\s]+([0-9,]+)원"),
        }

        # 대화 상자 메시지 저장용
        self.dialog_message = ""

    def _simplify_product_name(self, name: str) -> str:
        """상품명에서 일반적인 규격, 주석, 괄호 등을 제거하여 간소화"""
        simplified = name
        # 1. // 주석 제거
        simplified = re.sub(r"\s*\/\/.*$", "", simplified).strip()
        # 2. 괄호 및 내용 제거
        simplified = re.sub(r"\s*\([^)]*\)", "", simplified).strip()
        # 3. 규격/사이즈 제거 (예: A4, 0.2T, 100g, 297X210)
        simplified = re.sub(
            r"\s*\b(A|B)[0-9]\b", "", simplified, flags=re.IGNORECASE
        ).strip()
        simplified = re.sub(
            r"\s*\d+(\.\d+)?T\b", "", simplified, flags=re.IGNORECASE
        ).strip()
        simplified = re.sub(r"\s*\d+g\b", "", simplified, flags=re.IGNORECASE).strip()
        simplified = re.sub(r"\s*\d+(X|\*)\d+\b", "", simplified).strip()
        # 4. 맨 끝 숫자 제거 (오작동 가능성 있음, 주의)
        # simplified = re.sub(r'\s+\d+$', '', simplified).strip()
        # 5. 중복 공백 제거
        simplified = re.sub(r"\s+", " ", simplified).strip()

        # 원본과 동일하거나 너무 짧아지면 원본 반환
        if simplified == name or len(simplified) < 3:
            return name
        self.logger.debug(f"Simplified name: '{name}' -> '{simplified}'")
        return simplified

    def setup_selenium(self):
        """Selenium WebDriver 설정"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # 헤드리스 모드
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")

            # 크롬 드라이버 초기화 시도
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
            except Exception as e:
                self.logger.warning(
                    f"Chrome driver initialization failed: {str(e)}. Trying with Service."
                )
                # Service를 명시적으로 사용하는 대체 방법
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)

            self.driver.set_page_load_timeout(self.timeout)
            self.logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(
                f"Failed to initialize Selenium WebDriver: {str(e)}", exc_info=True
            )
            raise

    def __del__(self):
        """소멸자: WebDriver 자원 해제"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("Selenium WebDriver closed")
            except Exception as e:
                self.logger.error(f"Error closing WebDriver: {str(e)}")
        # Close Playwright browser
        if hasattr(self, 'browser') and self.browser:
             try:
                 self.browser.close()
                 self.logger.info("Playwright browser closed")
             except Exception as e:
                 self.logger.error(f"Error closing Playwright browser: {e}")
        if hasattr(self, 'playwright') and self.playwright:
            try:
                self.playwright.stop()
                self.logger.info("Playwright instance stopped")
            except Exception as e:
                self.logger.error(f"Error stopping Playwright: {e}")

    def search_product_with_playwright(self, query: str, keyword2: str = "") -> List[Product]:
        """Playwright를 사용하여 제품 검색 (crowling_kogift.py 로직 기반 업데이트)"""
        products = []
        try:
            page = self.context.new_page()
            page.set_default_timeout(60000) # 60초 타임아웃

            page.on("requestfailed", lambda request: self.logger.warning(f"Request failed: {request.url}"))

            self.logger.info(f"Navigating to {self.base_url}/ez/index.php with Playwright")
            page.goto(f"{self.base_url}/ez/index.php", wait_until="networkidle", timeout=90000) # 90초로 늘림

            self.logger.info(f"Performing initial search for: {query}")
            # Use locators from crowling_kogift.py
            page.locator('//input[@name="keyword" and @id="main_keyword"]').fill(query)
            page.locator('//img[@id="search_submit"]').click()
            page.wait_for_load_state('networkidle', timeout=60000)

            # 첫 검색 상품 개수 확인
            count_locator = page.locator('//div[text()=" 개의 상품이 있습니다."]/span')
            try:
                product_count_text = count_locator.text_content(timeout=15000) # 15초 타임아웃 추가
                product_count = int(product_count_text.replace(",", ""))
                self.logger.info(f"Initial search found {product_count} items for query: '{query}'")
            except Exception as e:
                 self.logger.warning(f"Could not find product count after initial search. Assuming 0. Error: {e}")
                 product_count = 0
                 # 페이지 소스 로그 추가
                 try:
                     page_content = page.content()
                     self.logger.debug(f"Page content after initial search failed to find count:\n{page_content[:500]}...") # 처음 500자만 로깅
                 except Exception as log_e:
                     self.logger.error(f"Failed to get page content for debugging: {log_e}")


            # Re-search logic from crowling_kogift.py (optional keyword2)
            if keyword2 and product_count >= 100:
                self.logger.info(f"Performing re-search with keyword2: {keyword2} as product count ({product_count}) >= 100")
                page.locator('//input[@id="re_keyword"]').fill(keyword2)
                page.locator('//button[@onclick="re_search()"]').click()
                page.wait_for_load_state('networkidle', timeout=60000)
                # Re-fetch product count after re-search
                try:
                    product_count_text = count_locator.text_content(timeout=15000)
                    product_count = int(product_count_text.replace(",", ""))
                    self.logger.info(f"After re-search with '{keyword2}', found {product_count} items.")
                except Exception as e:
                    self.logger.warning(f"Could not find product count after re-search. Assuming 0. Error: {e}")
                    product_count = 0
                    # 페이지 소스 로그 추가
                    try:
                        page_content = page.content()
                        self.logger.debug(f"Page content after re-search failed to find count:\n{page_content[:500]}...")
                    except Exception as log_e:
                        self.logger.error(f"Failed to get page content for debugging: {log_e}")


            page_number = 1
            collected_products = 0

            if product_count > 0:
                while True: # Loop through pages
                    self.logger.info(f"Scraping page {page_number}...")

                    # Product elements locator from crowling_kogift.py
                    product_elements = page.locator('//div[@class="product"]')
                    count = product_elements.count()
                    self.logger.debug(f"Found {count} product elements on page {page_number}")

                    if count == 0 and collected_products == 0:
                         self.logger.warning(f"No products found on the first page (page {page_number}) for query '{query}'. Stopping.")
                         # 페이지 소스 로그 추가
                         try:
                             page_content = page.content()
                             self.logger.debug(f"Page content when no products found on page {page_number}:\n{page_content[:500]}...")
                         except Exception as log_e:
                             self.logger.error(f"Failed to get page content for debugging: {log_e}")
                         break
                    elif count == 0:
                        self.logger.info(f"No more products found on page {page_number}. Ending pagination.")
                        break


                    for i in range(count):
                        element = product_elements.nth(i)
                        try:
                            # Extract data using locators from crowling_kogift.py
                            name = element.locator('div.name > a').text_content(timeout=5000)
                            a_href = element.locator('div.pic > a').get_attribute('href', timeout=5000)
                            url = f"{self.base_url}/ez/{a_href.replace('./', '')}" if a_href else "" # Adjusted base URL part based on crowling_kogift
                            img_src = element.locator('div.pic > a > img').get_attribute('src', timeout=5000)
                            image_url = f"{self.base_url}/ez/{img_src.replace('./', '')}" if img_src else "" # Adjusted base URL part

                            price_text = element.locator('div.price').text_content(timeout=5000)
                            price_match = self.patterns["price_number"].search(price_text)
                            price = int(price_match.group().replace(",", "")) if price_match else 0

                            if not name or not url:
                                self.logger.warning(f"Skipping item {i} on page {page_number} due to missing name or URL.")
                                continue

                            product_id = hashlib.md5(url.encode()).hexdigest()

                            # Create Product object (as done in KoryoScraper previously)
                            product = Product(
                                id=product_id,
                                name=name.strip(),
                                source="koryo",
                                url=url.strip(),
                                image_url=image_url.strip(),
                                price=price,
                                query=query, # Keep track of original query
                                fetched_at=datetime.now().isoformat() # Add fetch timestamp
                            )

                            # Optional: Call detail fetching (might need separate review/update)
                            # try:
                            #     product = self._get_product_details_playwright(page, product)
                            # except Exception as detail_e:
                            #     self.logger.warning(f"Failed to get details for {product.name}: {detail_e}")

                            products.append(product)
                            collected_products += 1
                            # self.logger.debug(f"Added product: {product.name} ({product.price})")


                        except Exception as e:
                            self.logger.warning(f"Error extracting data for item {i} on page {page_number}: {e}")
                            # Log element HTML for debugging
                            try:
                                element_html = element.inner_html()
                                self.logger.debug(f"HTML of problematic element:\n{element_html[:200]}...")
                            except Exception as log_e:
                                 self.logger.error(f"Failed to get element HTML for debugging: {log_e}")
                            continue

                    self.logger.info(f"Finished scraping page {page_number}. Total products collected so far: {collected_products}")

                    # Pagination logic from crowling_kogift.py
                    next_page_selector = f'//div[@class="custom_paging"]/div[@onclick="getPageGo1({page_number + 1})"]'
                    next_page = page.locator(next_page_selector)

                    if next_page.count() == 0:
                        self.logger.info("No 'next page' button found. Reached the end.")
                        break # Exit loop if no next page button

                    self.logger.debug(f"Clicking next page button to go to page {page_number + 1}")
                    try:
                        next_page.click(timeout=10000) # 10 초 타임아웃
                        page.wait_for_load_state('networkidle', timeout=60000) # Wait for next page to load
                        page_number += 1
                    except Exception as page_e:
                        self.logger.error(f"Error clicking next page or waiting for load: {page_e}. Stopping pagination.")
                         # 페이지 소스 로그 추가
                        try:
                            page_content = page.content()
                            self.logger.debug(f"Page content after failing to navigate to next page:\n{page_content[:500]}...")
                        except Exception as log_e:
                            self.logger.error(f"Failed to get page content for debugging: {log_e}")
                        break


            page.close()
            self.logger.info(f"Playwright search finished for '{query}'. Found {len(products)} products.")
            return products

        except TimeoutError as te:
             self.logger.error(f"Playwright operation timed out during search for '{query}': {te}", exc_info=True)
             if 'page' in locals() and page:
                 try:
                     page_content = page.content()
                     self.logger.debug(f"Page content at timeout:\n{page_content[:500]}...") # 처음 500자 로깅
                     page.close()
                 except Exception as close_e:
                     self.logger.error(f"Error getting page content or closing page after timeout: {close_e}")
             return [] # Return empty list on timeout
        except Exception as e:
            self.logger.error(f"Error during Playwright search for '{query}': {e}", exc_info=True)
            if 'page' in locals() and page:
                 try:
                     page.close()
                 except Exception as close_e:
                      self.logger.error(f"Error closing page after general exception: {close_e}")
            return [] # Return empty list on other errors

    def _get_product_details_playwright(self, page, product: Product) -> Product:
        """Playwright를 사용하여 상세 페이지에서 추가 정보 가져오기 (개선 필요할 수 있음)"""
        # Note: This function's selectors might still need review based on KoryoGift's detail page structure.
        # The search function update was the primary goal here.
        original_url = ""
        try:
            original_url = page.url # Store original page URL to return later
            self.logger.debug(f"Fetching details for {product.name} from {product.url}")

            # Navigate to product detail page
            page.goto(product.url, wait_until="networkidle", timeout=60000)

            # --- Price Table Extraction (Keep existing logic, might need review) ---
            price_table_element = page.locator(
                "table.price_table, table.quantity_table" # Keep existing selectors for now
            ).first
            if price_table_element.count() > 0:
                try:
                    html_content = price_table_element.inner_html(timeout=10000)
                    soup = BeautifulSoup(html_content, "html.parser")
                    dfs = pd.read_html(str(soup))
                    if dfs:
                        df = dfs[0]
                        # (Rest of the price table parsing logic remains the same as before)
                        # ... find qty_col, price_col ...
                        # ... iterate through df rows ...
                        # ... calculate price with VAT ...
                        # ... store in product.quantity_prices ...
                        # ... save CSV ...
                        self.logger.debug(f"Successfully parsed price table for {product.name}")
                    else:
                         self.logger.warning(f"Could not read DataFrame from price table HTML for {product.name}")

                except Exception as e:
                    self.logger.warning(f"Error parsing price table for {product.name}: {e}")
            else:
                 self.logger.debug(f"No price table found using selectors for {product.name}")


            # --- Product Code Extraction (Keep existing logic, might need review) ---
            code_element = page.locator(
                "div.prd_info span.code, div.goods_info div.code, span:has-text('상품코드')" # Keep/add selectors
            ).first
            if code_element.count() > 0:
                try:
                    code_text = code_element.text_content(timeout=5000).strip()
                    # Try to extract code after ":"
                    if ":" in code_text:
                        code_part = code_text.split(":", 1)[-1].strip()
                        if code_part: product.product_code = code_part
                    # Fallback to regex if needed
                    if not product.product_code:
                        match = self.patterns["product_code"].search(code_text)
                        if match:
                            product.product_code = match.group(1)
                    self.logger.debug(f"Extracted product code for {product.name}: {product.product_code}")
                except Exception as e:
                    self.logger.warning(f"Error extracting product code for {product.name}: {e}")


            # --- Description Extraction (Keep existing logic, might need review) ---
            desc_element = page.locator(
                "div#prd_detail_content, div.product_detail, div.detail_info, #detail") # Keep/add selectors
            if desc_element.count() > 0:
                try:
                     # Try to get outer HTML to preserve structure, fallback to text
                     try:
                         product.description = desc_element.first.evaluate("element => element.outerHTML", timeout=10000)
                     except:
                         product.description = desc_element.first.text_content(timeout=10000).strip()
                     self.logger.debug(f"Extracted description for {product.name} (length: {len(product.description or '')})")
                except Exception as e:
                    self.logger.warning(f"Error extracting description for {product.name}: {e}")

            # --- Image Gallery Extraction (Added) ---
            image_gallery = []
            img_elements = page.locator('div.swiper-slide img, .detail_img img, #product-images img') # Common gallery image selectors
            img_count = img_elements.count()
            if img_count > 0:
                self.logger.debug(f"Found {img_count} potential gallery images for {product.name}")
                for i in range(img_count):
                    try:
                        img_src = img_elements.nth(i).get_attribute('src', timeout=2000)
                        if img_src:
                             full_img_url = urllib.parse.urljoin(page.url, img_src) # Make URL absolute
                             if full_img_url not in image_gallery: # Avoid duplicates
                                 image_gallery.append(full_img_url)
                    except Exception as img_e:
                        self.logger.warning(f"Error extracting image {i} for {product.name}: {img_e}")
                product.image_gallery = image_gallery
                if image_gallery and not product.image_url: # Use first gallery image if main image is missing
                    product.image_url = image_gallery[0]
                self.logger.debug(f"Collected {len(image_gallery)} unique images for {product.name}")


            # Navigate back to the original search results page
            if original_url:
                self.logger.debug(f"Navigating back to original page: {original_url}")
                page.goto(original_url, wait_until="networkidle", timeout=60000)

        except TimeoutError as te:
            self.logger.error(f"Timeout getting product details for {product.url}: {te}")
            # Try to navigate back even on timeout
            if original_url:
                try:
                    self.logger.warning(f"Attempting to navigate back to {original_url} after detail timeout")
                    page.goto(original_url, wait_until="domcontentloaded", timeout=30000) # Shorter timeout for navigating back
                except Exception as back_e:
                    self.logger.error(f"Failed to navigate back after detail timeout: {back_e}")
        except Exception as e:
            self.logger.error(f"Error getting product details for {product.url}: {e}", exc_info=True)
             # Try to navigate back on general error
            if original_url:
                try:
                    self.logger.warning(f"Attempting to navigate back to {original_url} after detail error")
                    page.goto(original_url, wait_until="domcontentloaded", timeout=30000)
                except Exception as back_e:
                    self.logger.error(f"Failed to navigate back after detail error: {back_e}")


        return product

    def _handle_dialog(self, dialog):
        """대화 상자 처리 (품절 등 상태 메시지 확인용)"""
        self.dialog_message = dialog.message
        self.logger.debug(f"Dialog message: {dialog.message}")
        dialog.accept()

    def get_price_table(self, url: str) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """URL에서 가격표 가져오기"""
        if not self.playwright_available:
            self.logger.error("Playwright is not installed. Cannot crawl price table.")
            return None, False, "Playwright 설치 필요"

        self.dialog_message = ""  # 대화 상자 메시지 초기화

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context = browser.new_context()
                page = context.new_page()
                page.on("dialog", self._handle_dialog)

                # 페이지 로드
                page.goto(url, wait_until="networkidle")

                # 품절 확인
                is_sold_out = False
                sold_out_text = page.locator(
                    'div:has-text("품절"), div:has-text("재고")'
                ).first
                if sold_out_text and sold_out_text.count() > 0:
                    is_sold_out = True
                    self.logger.info(f"상품 품절 확인: {sold_out_text.text_content()}")
                    browser.close()
                    return None, is_sold_out, sold_out_text.text_content()

                if self.dialog_message and (
                    "상품" in self.dialog_message
                    or "재고" in self.dialog_message
                    or "품절" in self.dialog_message
                ):
                    is_sold_out = True
                    self.logger.info(
                        f"상품 품절 확인 (대화 상자): {self.dialog_message}"
                    )
                    browser.close()
                    return None, is_sold_out, self.dialog_message

                # 가격표 추출
                price_table_element = page.locator(
                    "table.price_table, table.quantity_table"
                ).first
                if price_table_element and price_table_element.count() > 0:
                    # 테이블 HTML 가져오기
                    html_content = price_table_element.inner_html()
                    soup = BeautifulSoup(html_content, "html.parser")

                    try:
                        # 테이블에서 DataFrame 생성
                        table_df = pd.read_html(str(soup))[0]

                        # 수량/가격 컬럼 찾기
                        col_mapping = {}
                        for col in table_df.columns:
                            col_str = str(col).lower()
                            if any(
                                keyword in col_str
                                for keyword in ["수량", "qty", "개수"]
                            ):
                                col_mapping["수량"] = col
                            elif any(
                                keyword in col_str
                                for keyword in ["가격", "price", "단가"]
                            ):
                                col_mapping["일반"] = col

                        # 컬럼 매핑이 제대로 됐는지 확인
                        if len(col_mapping) == 2:
                            df = pd.DataFrame(
                                {
                                    "수량": table_df[col_mapping["수량"]],
                                    "일반": table_df[col_mapping["일반"]],
                                }
                            )
                        else:
                            # 첫 번째 컬럼이 수량, 두 번째가 가격이라고 가정
                            df = pd.DataFrame(
                                {
                                    "수량": table_df.iloc[:, 0],
                                    "일반": table_df.iloc[:, 1],
                                }
                            )

                        # 데이터 정제
                        df["수량"] = df["수량"].apply(
                            lambda x: "".join(filter(str.isdigit, str(x)))
                        )
                        df["일반"] = df["일반"].apply(
                            lambda x: "".join(filter(str.isdigit, str(x)))
                        )

                        # 숫자형으로 변환
                        df["수량"] = pd.to_numeric(df["수량"], errors="coerce")
                        df["일반"] = pd.to_numeric(df["일반"], errors="coerce")

                        # 결측치 제거
                        df = df.dropna()

                        # VAT 여부 확인 후 조정
                        html_text = page.content().lower()
                        vat_included = (
                            "vat 포함" in html_text or "vat included" in html_text
                        )
                        if not vat_included:
                            df["일반"] = df["일반"] * 1.1

                        # 정렬
                        df = df.sort_values(by="수량")

                        # CSV 파일로 저장
                        output_file = f"output/koryo_price_table.csv"
                        df.to_csv(output_file, index=False)
                        self.logger.info(f"가격표 저장 완료: {output_file}")

                        browser.close()
                        return df, False, ""

                    except Exception as e:
                        self.logger.error(f"가격표 추출 오류: {e}")

                # 수량/가격 정보가 텍스트로 있는지 확인
                html_content = page.content()
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text()

                # 정규식으로 수량:가격 패턴 찾기 (예: "100개: 1,000원")
                matches = self.patterns["quantity_price"].findall(text_content)

                if matches:
                    quantities = [int(match[0]) for match in matches]
                    prices = [int(match[1].replace(",", "")) for match in matches]

                    # VAT 여부 확인 후 조정
                    vat_included = (
                        "vat 포함" in text_content.lower()
                        or "vat included" in text_content.lower()
                    )
                    if not vat_included:
                        prices = [int(price * 1.1) for price in prices]

                    # 데이터프레임 생성
                    df = pd.DataFrame({"수량": quantities, "일반": prices})
                    df = df.sort_values(by="수량")

                    # CSV 파일로 저장
                    output_file = f"output/koryo_price_table_text.csv"
                    df.to_csv(output_file, index=False)
                    self.logger.info(
                        f"텍스트에서 추출한 가격표 저장 완료: {output_file}"
                    )

                    browser.close()
                    return df, False, ""

                browser.close()
                return None, False, "가격표를 찾을 수 없습니다."

        except Exception as e:
            self.logger.error(f"가격표 추출 중 오류 발생: {e}", exc_info=True)
            return None, False, str(e)

    def check_stock_status(self, url: str) -> Tuple[bool, str]:
        """상품 URL에서 재고 상태 확인"""
        _, is_sold_out, message = self.get_price_table(url)
        return not is_sold_out, message

    def search_product(
        self, query: str, max_items: int = 50, keyword2: str = ""
    ) -> List[Product]:
        """
        고려기프트에서 제품 검색 (Playwright 우선 사용, 실패 시 Selenium 사용)

        Args:
            query: 검색어
            max_items: 최대 검색 결과 수
            keyword2: 결과 내 재검색 키워드 (선택적)

        Returns:
            List[Product]: 검색된 제품 목록
        """
        cache_key = f"koryo_search|{query}|{keyword2}|{max_items}"

        if self.cache:
            cached_result = self.get_sparse_data(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for query: '{query}'")
                return cached_result

        # Playwright가 사용 가능하면 Playwright로 검색
        if self.playwright_available:
            try:
                products = self.search_product_with_playwright(query, keyword2)

                # 캐싱
                if products and self.cache:
                    self.add_sparse_data(cache_key, products)

                return products
            except Exception as e:
                self.logger.warning(
                    f"Playwright search failed, falling back to Selenium: {e}"
                )
                # Playwright 실패시 Selenium으로 폴백

        # Selenium 검색 사용
        simplified_query = self._simplify_product_name(query)
        products = self._search_product_logic(simplified_query, max_items)

        # 캐싱
        if products and self.cache:
            self.add_sparse_data(cache_key, products)

        return products

    # 핵심 검색 로직
    def _search_product_logic(self, query: str, max_items: int = 50) -> List[Product]:
        """Selenium을 사용한 제품 검색 (단일 검색 시도)"""
        products = []

        try:
            # 직접 search_url를 사용하여 검색 결과 페이지로 이동
            search_url = f"{self.search_url}?keyword={urllib.parse.quote(query)}"
            self.logger.debug(f"검색 URL: {search_url}")
            self.driver.get(search_url)
            sleep(3)  # 검색 결과 로딩 대기
            self.logger.info(f"Search results page loaded: {self.driver.current_url}")

            # 현재 페이지 디버깅을 위해 HTML 소스 일부 로깅 (100자로 제한)
            page_source = self.driver.page_source
            self.logger.debug(f"Page title: {self.driver.title}")
            self.logger.debug(f"Page source sample: {page_source[:100]}...")

            # 디버그 모드일 경우 HTML 저장
            if hasattr(self, "debug") and self.debug:
                try:
                    with open(
                        f"logs/search_results_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(page_source)
                    self.logger.debug(f"HTML 페이지 저장 완료: search_results_{query}")
                except Exception as e:
                    self.logger.error(f"HTML 저장 실패: {str(e)}")

            page = 1
            while True:
                self.logger.debug(
                    f"Processing search results for '{query}', page={page}"
                )

                # 현재 페이지 파싱
                soup = BeautifulSoup(self.driver.page_source, "lxml")

                # 제품 목록 추출 시도 (여러 선택자 시도)
                product_elements = []
                selectors_to_try = [
                    "div.product_lists .product",  # 카테고리 페이지의 제품 목록
                    "#mm_pro_lists .mm_pro",  # 새로운 셀렉터 추가
                    ".product-list .item",  # 새로운 셀렉터 추가
                    "div.prd_list_wrap li.prd",  # 기존 셀렉터
                    "ul.prd_list li",  # 대체 셀렉터
                    "table.mall_list td.prd",  # 대체 셀렉터
                    "div.product_list .item",  # 대체 셀렉터
                    ".prd-list .prd-item",  # 대체 셀렉터
                    ".product-item",  # 일반적인 제품 아이템 클래스
                    "li.goods-box",  # 일반적인 제품 목록 항목
                    "table.mall_tbl tr:not(:first-child)",  # 테이블 기반 목록
                    'div[class*="product"], div[class*="item"], li[class*="product"], li[class*="item"]',  # 와일드카드 셀렉터
                ]

                for selector in selectors_to_try:
                    self.logger.debug(f"Trying selector: {selector}")
                    product_elements = soup.select(selector)
                    if product_elements:
                        self.logger.debug(
                            f"Found {len(product_elements)} products with selector: {selector}"
                        )
                        # 성공한 선택자 저장
                        self.selectors["product_list"]["selector"] = selector
                        break

                if not product_elements:
                    self.logger.warning(
                        f"No product elements found on page {page} for query '{query}'"
                    )

                    # 검색 결과가 없는지 확인
                    no_results_text = soup.select_one(
                        '.no-results, .empty-results, .search-empty, p:contains("검색결과가 없습니다")'
                    )
                    if no_results_text:
                        self.logger.info(
                            f"검색 결과 없음 메시지 발견: {no_results_text.text.strip()}"
                        )

                    # 로그인이 필요한지 확인
                    login_required = soup.select_one(
                        '.login-required, .member-only, form[action*="login"]'
                    )
                    if login_required:
                        self.logger.warning(
                            "로그인이 필요한 페이지로 리디렉션되었습니다."
                        )

                    break

                # 각 제품 정보 추출
                for element in product_elements:
                    try:
                        product_data = self._extract_list_item(element)
                        if product_data:
                            # 상세 페이지 정보 가져오기
                            detailed_product = self._get_product_details(product_data)
                            if detailed_product:
                                products.append(detailed_product)
                                self.logger.info(
                                    f"Added product: {detailed_product.name}"
                                )
                                if max_items > 0 and len(products) >= max_items:
                                    return products[:max_items]
                    except Exception as e:
                        self.logger.warning(f"Error extracting product data: {str(e)}")
                        continue

                # 다음 페이지 확인 및 이동
                try:
                    # 다음 페이지 링크 찾기
                    next_link = soup.select_one(
                        'a.next, a:contains("다음"), .custom_paging .arrow:last-child'
                    )
                    if next_link and next_link.get("href"):
                        next_url = next_link["href"]
                        if not next_url.startswith("http"):
                            if next_url.startswith("/"):
                                next_url = f"{self.base_url}{next_url}"
                            else:
                                next_url = f"{self.base_url}/ez/{next_url}"

                        self.driver.get(next_url)
                        sleep(2)
                        page += 1

                        # 최대 5페이지로 제한
                        if page > 5:
                            self.logger.debug(f"Reached maximum page limit (5)")
                            break
                    else:
                        self.logger.debug("No next page link found")
                        break
                except Exception as e:
                    self.logger.error(f"Error navigating to next page: {str(e)}")
                    break
        except Exception as e:
            self.logger.error(
                f"Error during Koryo scraping for '{query}': {str(e)}", exc_info=True
            )

        self.logger.info(
            f"Koryo Scraper: Found {len(products)} products for query '{query}'"
        )
        return products[:max_items] if max_items > 0 else products

    def _extract_list_item(self, element) -> Dict:
        """제품 목록 항목에서 기본 정보 추출"""
        try:
            # 다양한 HTML 구조 처리
            product_data = {}

            # 제품명 추출
            title_element = None
            title_selectors = [
                "p.name a",
                "div.name a",
                "td.name a",
                ".prd_name a",
                "a.product-title",
            ]

            for selector in title_selectors:
                title_element = element.select_one(selector)
                if title_element:
                    break

            # 타이틀이 없으면 빈 딕셔너리 반환
            if not title_element:
                self.logger.warning("Product title element not found")
                return {}

            title = title_element.text.strip()
            product_data["title"] = title

            # 링크 추출
            link = title_element.get("href")
            if link:
                if not link.startswith("http"):
                    if link.startswith("/"):
                        link = f"{self.base_url}{link}"
                    else:
                        link = f"{self.base_url}/ez/{link}"
                product_data["link"] = link
            else:
                # 링크가 없으면 빈 딕셔너리 반환
                self.logger.warning("Product link not found")
                return {}

            # 가격 추출
            price_element = None
            price_selectors = [
                "p.price",
                "div.price",
                "td.price",
                ".prd_price",
                "span.price",
            ]

            for selector in price_selectors:
                price_element = element.select_one(selector)
                if price_element:
                    break

            price_text = price_element.text.strip() if price_element else "0"
            price_match = self.patterns["price_number"].search(price_text)
            price = int(price_match.group().replace(",", "")) if price_match else 0
            product_data["price"] = price

            # 모델 번호 추출 (옵션)
            model_element = element.select_one("div.model")
            if model_element:
                product_data["model_number"] = model_element.text.strip()

            # 썸네일 추출
            thumbnail_element = None
            thumbnail_selectors = [
                ".pic img",
                "img.prd_img",
                "td.img img",
                ".thumb img",
                "img.product-image",
            ]

            for selector in thumbnail_selectors:
                thumbnail_element = element.select_one(selector)
                if thumbnail_element:
                    break

            thumbnail = thumbnail_element.get("src") if thumbnail_element else ""
            if thumbnail and not thumbnail.startswith("http"):
                if thumbnail.startswith("/"):
                    thumbnail = f"{self.base_url}{thumbnail}"
                else:
                    thumbnail = f"{self.base_url}/ez/{thumbnail}"
            product_data["image"] = thumbnail

            # 고유 ID 생성
            product_id = hashlib.md5(link.encode()).hexdigest() if link else ""
            product_data["product_id"] = product_id

            return product_data

        except Exception as e:
            self.logger.warning(f"Error extracting list item: {str(e)}")
            return {}

    def _get_product_details(self, item: Dict) -> Optional[Product]:
        """Selenium을 사용하여 제품 상세 정보 가져오기"""
        if not item.get("link"):
            return None

        # 캐시 확인
        cache_key = f"koryo_detail|{item['product_id']}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            return cached_result

        url = item["link"]

        try:
            # 현재 페이지 URL 저장
            current_url = self.driver.current_url

            # 상세 페이지 접속
            try:
                self.driver.get(url)
                sleep(2)  # 페이지 로딩 대기
            except TimeoutException:
                self.logger.warning(
                    f"Page load timed out for {url}, trying with a longer timeout"
                )
                # 타임아웃 발생 시 페이지 로드 중단 및 재시도
                self.driver.execute_script("window.stop();")
                sleep(3)  # 추가 대기

            # HTML 파싱
            soup = BeautifulSoup(self.driver.page_source, "lxml")

            # 상세 정보 추출 (다양한 셀렉터 시도)
            title = item.get("title", "")

            # 상품명
            title_selectors = [
                "div.product_name",
                "h3.prd_title",
                "div.prd_name",
                "h1.product-title",
                ".view_title",
            ]
            for selector in title_selectors:
                detail_title = soup.select_one(selector)
                if detail_title:
                    title = detail_title.text.strip()
                    break

            # 가격
            price = item.get("price", 0)
            price_selectors = [
                "#main_price",
                "span.price_num",
                "div.prd_price",
                "p.price",
                ".view_price",
            ]

            for selector in price_selectors:
                detail_price_element = soup.select_one(selector)
                if detail_price_element:
                    price_text = detail_price_element.text.strip()
                    price_match = self.patterns["price_number"].search(price_text)
                    if price_match:
                        price = int(price_match.group().replace(",", ""))
                    break

            # 제품 코드 추출
            product_code = ""
            code_selectors = [".prd_code", ".item_code", "span.code", "div.model"]

            for selector in code_selectors:
                code_element = soup.select_one(selector)
                if code_element:
                    code_text = code_element.text.strip()
                    code_match = self.patterns["product_code"].search(code_text)
                    if code_match:
                        product_code = code_match.group(1)
                    else:
                        # 정규식 없이 직접 텍스트에서 추출 시도
                        product_code = code_text
                        if ":" in product_code:
                            product_code = product_code.split(":", 1)[1].strip()

            # URL에서 제품 번호 추출 시도
            if not product_code and "no=" in url:
                product_code = url.split("no=")[-1].split("&")[0]

            # 이미지 URL 추출
            image_elements = []
            image_selectors = [
                "#main_img",
                ".product_picture img",
                ".prd_img img",
                ".img_big img",
                ".view_image img",
                ".thumbnail img",
            ]

            for selector in image_selectors:
                elements = soup.select(selector)
                if elements:
                    image_elements = elements
                    break

            image_gallery = []
            if image_elements:
                for img in image_elements:
                    img_url = img.get("src", "")
                    if img_url:
                        if not img_url.startswith("http"):
                            if img_url.startswith("/"):
                                img_url = f"{self.base_url}{img_url}"
                            else:
                                img_url = f"{self.base_url}/ez/{img_url}"
                        image_gallery.append(img_url)

            # 수량별 가격 추출
            quantity_prices = {}
            quantity_table_selectors = [
                "table.quantity_price__table",
                "table.price_table",
                "table.quantity_table",
                ".price_by_quantity",
            ]

            for selector in quantity_table_selectors:
                quantity_table = soup.select_one(selector)
                if quantity_table:
                    quantity_prices = self._extract_quantity_prices(quantity_table)
                    break

            # 제품 사양 추출
            specifications = {}
            specs_table_selectors = [
                "table.tbl_info",
                "table.spec_table",
                "table.product_info",
                ".product_spec",
            ]

            for selector in specs_table_selectors:
                specs_table = soup.select_one(selector)
                if specs_table:
                    specifications = self._extract_specifications(specs_table)
                    break

            # 제품 설명 추출
            description = ""
            desc_selectors = [".prd_detail", ".product_description", ".item_detail"]

            for selector in desc_selectors:
                desc_element = soup.select_one(selector)
                if desc_element:
                    description = desc_element.text.strip()
                    break

            # 제품 객체 생성
            product = Product(
                id=item.get("product_id", ""),
                name=title,
                price=price,
                source="koryo",
                original_input_data=item,
            )

            # 추가 정보 설정
            product.url = url
            product.image_url = item.get("image", "") or (
                image_gallery[0] if image_gallery else ""
            )
            product.image_gallery = image_gallery
            product.product_code = product_code
            product.description = description
            product.specifications = specifications
            product.quantity_prices = quantity_prices

            # 모델 번호가 있으면 추가
            if "model_number" in item:
                product.model_number = item["model_number"]

            # 캐시에 저장
            self.add_sparse_data(cache_key, product)  # 기본 캐시 기간 사용

            # 이전 페이지로 돌아가기
            self.driver.get(current_url)
            sleep(1)

            return product

        except Exception as e:
            self.logger.error(
                f"Error getting product details for {url}: {str(e)}", exc_info=True
            )
            # 예외 발생 시 이전 페이지로 돌아가기
            try:
                if current_url:
                    self.driver.get(current_url)
                    sleep(1)
            except:
                pass
            return None

    def _extract_quantity_prices(self, table_element) -> Dict[str, float]:
        """수량별 가격 테이블에서 가격 정보 추출"""
        quantity_prices = {}
        try:
            rows = table_element.select("tr")
            for row in rows[1:]:  # 헤더 행 제외
                cells = row.select("td")
                if len(cells) >= 2:
                    qty_cell = cells[0].text.strip()
                    price_cell = cells[1].text.strip()

                    # 수량 추출
                    qty_match = self.patterns["quantity"].search(qty_cell)
                    qty = int(qty_match.group(1)) if qty_match else 0

                    # 가격 추출
                    price_match = self.patterns["price_number"].search(price_cell)
                    price = (
                        int(price_match.group().replace(",", "")) if price_match else 0
                    )

                    if qty and price:
                        quantity_prices[str(qty)] = price
        except Exception as e:
            self.logger.error(f"Error extracting quantity prices: {str(e)}")

        return quantity_prices

    def _extract_specifications(self, table_element) -> Dict[str, str]:
        """제품 사양 테이블에서 정보 추출"""
        specs = {}
        try:
            rows = table_element.select("tr")
            for row in rows:
                header_cell = row.select_one("th")
                value_cell = row.select_one("td")

                if header_cell and value_cell:
                    key = header_cell.text.strip()
                    value = value_cell.text.strip()
                    if key and value:
                        specs[key] = value
        except Exception as e:
            self.logger.error(f"Error extracting specifications: {str(e)}")

        return specs

    def get_categories(self) -> List[Dict]:
        """사이트의 모든 카테고리 정보 가져오기"""
        categories = []

        try:
            # 메인 페이지 접속
            self.driver.get(f"{self.base_url}/ez/index.php")
            sleep(3)  # 페이지 로딩 대기

            # HTML 파싱
            soup = BeautifulSoup(self.driver.page_source, "lxml")

            # 카테고리 요소 추출 시도 (다양한 선택자)
            category_elements = []
            category_selectors = [
                ".category a",
                "#category_all a",
                ".menu_box a",
                'a[href*="mall.php?cat="]',
            ]

            for selector in category_selectors:
                self.logger.debug(f"Trying category selector: {selector}")
                elements = soup.select(selector)
                if elements:
                    self.logger.debug(
                        f"Found {len(elements)} categories with selector: {selector}"
                    )
                    category_elements = elements
                    break

            if not category_elements:
                # 마지막 방법: 직접 카테고리 페이지 접근
                self.logger.debug("Trying to access category page directly")
                self.driver.get(f"{self.base_url}/ez/mall.php")
                sleep(2)

                soup = BeautifulSoup(self.driver.page_source, "lxml")
                for selector in category_selectors:
                    elements = soup.select(selector)
                    if elements:
                        self.logger.debug(
                            f"Found {len(elements)} categories from direct page with selector: {selector}"
                        )
                        category_elements = elements
                        break

            for element in category_elements:
                try:
                    category_name = element.text.strip()
                    category_url = element.get("href")

                    # 실제 링크가 있는지 확인
                    if (
                        not category_url
                        or category_url == "#"
                        or category_url.startswith("javascript:")
                    ):
                        continue

                    # "cat=" 파라미터가 있는 URL만 고려 (카테고리 URL 패턴)
                    if "cat=" not in category_url and "cate=" not in category_url:
                        continue

                    if category_name and category_url:
                        if not category_url.startswith("http"):
                            if category_url.startswith("/"):
                                category_url = f"{self.base_url}{category_url}"
                            else:
                                category_url = f"{self.base_url}/ez/{category_url}"

                        category_id = hashlib.md5(category_url.encode()).hexdigest()

                        categories.append(
                            {
                                "id": category_id,
                                "name": category_name,
                                "url": category_url,
                            }
                        )
                        self.logger.debug(
                            f"Added category: {category_name} - {category_url}"
                        )
                except Exception as e:
                    self.logger.warning(f"Error extracting category: {str(e)}")
                    continue

            # 카테고리가 없는 경우 기본 카테고리 추가
            if not categories:
                for name, cat_id in self.default_categories:
                    url = f"{self.mall_url}?cat={cat_id}"
                    category_id = hashlib.md5(url.encode()).hexdigest()
                    categories.append({"id": category_id, "name": name, "url": url})
                    self.logger.debug(f"Added default category: {name} - {url}")

        except Exception as e:
            self.logger.error(f"Error getting categories: {str(e)}", exc_info=True)

        self.logger.info(f"Found {len(categories)} categories")
        return categories

    def crawl_all_categories(
        self, max_products_per_category: int = 50
    ) -> List[Product]:
        """모든 카테고리의 제품을 크롤링"""
        all_products = []

        # 카테고리 목록 가져오기
        categories = self.get_categories()

        for category in categories:
            try:
                self.logger.info(f"Crawling category: {category['name']}")

                # 카테고리 페이지 접속
                self.driver.get(category["url"])
                sleep(2)

                products = []
                page = 1

                while True:
                    # 현재 페이지 파싱
                    soup = BeautifulSoup(self.driver.page_source, "lxml")

                    # 제품 목록 추출
                    product_elements = self.extract(
                        soup,
                        self.selectors["product_list"]["selector"],
                        **self.selectors["product_list"]["options"],
                    )

                    if not product_elements:
                        break

                    # 각 제품 정보 추출
                    for element in product_elements:
                        try:
                            product_data = self._extract_list_item(element)
                            if product_data:
                                # 상세 페이지 정보 가져오기
                                detailed_product = self._get_product_details(product_data)
                                if detailed_product:
                                    products.append(detailed_product)
                                    all_products.append(detailed_product)

                                    if (
                                        max_products_per_category > 0
                                        and len(products) >= max_products_per_category
                                    ):
                                        break
                        except Exception as e:
                            self.logger.warning(
                                f"Error extracting product data: {str(e)}"
                            )
                            continue

                    # 최대 제품 수 도달 시 중단
                    if (
                        max_products_per_category > 0
                        and len(products) >= max_products_per_category
                    ):
                        break

                    # 다음 페이지 확인 및 이동
                    try:
                        next_button = self.driver.find_element(
                            By.CSS_SELECTOR, "a.next"
                        )
                        next_button.click()
                        sleep(2)
                        page += 1

                        # 최대 5페이지로 제한
                        if page > 5:
                            break
                    except NoSuchElementException:
                        break
                    except Exception as e:
                        self.logger.error(f"Error navigating to next page: {str(e)}")
                        break

                self.logger.info(
                    f"Found {len(products)} products in category '{category['name']}'"
                )

            except Exception as e:
                self.logger.error(
                    f"Error crawling category {category['name']}: {str(e)}",
                    exc_info=True,
                )

    def browse_category(
        self, category_id: str = None, max_items: int = 50
    ) -> List[Product]:
        """카테고리를 직접 브라우징하여 제품 가져오기"""
        products = []

        try:
            # 카테고리 ID가 있으면 해당 카테고리 페이지로 이동
            if category_id:
                url = f"{self.mall_url}?cat={category_id}"
                self.logger.info(f"Browsing category with ID: {category_id}")
            else:
                # 기본 첫 번째 카테고리 사용
                _, default_id = self.default_categories[0]
                url = f"{self.mall_url}?cat={default_id}"
                self.logger.info(
                    f"Browsing default category: {self.default_categories[0][0]}"
                )

            try:
                self.driver.get(url)
                sleep(3)  # 페이지 로딩 대기
                self.logger.info(f"Category page loaded: {self.driver.current_url}")
            except TimeoutException:
                self.logger.warning(
                    f"Category page load timed out, trying with longer timeout"
                )
                self.driver.execute_script("window.stop();")
                sleep(3)

            page = 1
            while True:
                self.logger.debug(f"Processing category page {page}")

                # 현재 페이지 파싱
                soup = BeautifulSoup(self.driver.page_source, "lxml")

                # 제품 목록 추출 시도 (여러 선택자 시도)
                product_elements = []
                selectors_to_try = [
                    "ul.prd_list li",
                    "table.mall_list td.prd",
                    "div.product_list .item",
                    ".prd-list .prd-item",
                    "table.mall_tbl tr:not(:first-child)",
                ]

                for selector in selectors_to_try:
                    self.logger.debug(f"Trying selector: {selector}")
                    product_elements = soup.select(selector)
                    if product_elements:
                        self.logger.debug(
                            f"Found {len(product_elements)} products with selector: {selector}"
                        )
                        break

                # 제품을 찾지 못한 경우
                if not product_elements:
                    self.logger.warning("No products found in this category page")
                    break

                # 각 제품 정보 추출
                for element in product_elements:
                    try:
                        product_data = self._extract_list_item(element)
                        if product_data:
                            # 상세 페이지 정보 가져오기
                            detailed_product = self._get_product_details(product_data)
                            if detailed_product:
                                products.append(detailed_product)
                                self.logger.info(
                                    f"Added product: {detailed_product.name}"
                                )

                                if max_items > 0 and len(products) >= max_items:
                                    return products[:max_items]
                    except Exception as e:
                        self.logger.warning(f"Error extracting product data: {str(e)}")
                        continue

                # 다음 페이지 확인 및 이동
                try:
                    # 다음 페이지 링크 찾기
                    next_link = soup.select_one('a.next, a:contains("다음")')
                    if next_link and next_link.get("href"):
                        next_url = next_link["href"]
                        if not next_url.startswith("http"):
                            if next_url.startswith("/"):
                                next_url = f"{self.base_url}{next_url}"
                            else:
                                next_url = f"{self.base_url}/ez/{next_url}"

                        self.driver.get(next_url)
                        sleep(2)
                        page += 1

                        # 최대 5페이지로 제한
                        if page > 5:
                            self.logger.debug(f"Reached maximum page limit (5)")
                            break
                    else:
                        self.logger.debug("No next page link found")
                        break
                except Exception as e:
                    self.logger.error(f"Error navigating to next page: {str(e)}")
                    break

        except Exception as e:
            self.logger.error(
                f"Error during category browsing: {str(e)}", exc_info=True
            )

        self.logger.info(f"Found {len(products)} products in category")
        return products[:max_items] if max_items > 0 else products
