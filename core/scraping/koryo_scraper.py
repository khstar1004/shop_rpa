"""
고려기프트 웹사이트 스크래퍼 모듈
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
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
from urllib.parse import urljoin

# 추가: Playwright 임포트 시도 (선택적 의존성)
try:
    from playwright.sync_api import TimeoutError, sync_playwright, Error as PlaywrightError

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Add imports for caching
from utils.caching import FileCache, cache_result

from ..data_models import Product
from .base_multi_layer_scraper import BaseMultiLayerScraper
from .extraction_strategy import DOMExtractionStrategy, TextExtractionStrategy
from .utils import extract_main_image


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

        # Define selector lists for product details
        self.title_selectors = [
            "h3.detail_tit",
            "h3.prd_title",
            ".view_title",
            ".product_name",
            "h1.product-title"
        ]
        
        self.price_selectors = [
            "dl.detail_price dd",
            ".price_num",
            "#main_price",
            ".product-price",
            ".price"
        ]
        
        self.code_selectors = [
            "div.product_code",
            ".product-code",
            ".item-code",
            "span.code"
        ]
        
        self.image_selectors = [
            "div.swiper-slide img",
            ".product-image img",
            ".detail_img img",
            ".product-gallery img"
        ]

        # New ordering: first create output directory and define headers
        os.makedirs("output", exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": self.base_url,
        }

        # Playwright available check remains, but initialization is removed
        self.playwright_available = PLAYWRIGHT_AVAILABLE
        if not self.playwright_available:
            self.logger.warning("Playwright is not available. Some features will be limited.")

        # Selenium 설정
        self.driver = None
        self.setup_selenium()

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
            
            # 추가된 옵션들
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--ignore-ssl-errors")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            
            # 네트워크 관련 설정
            chrome_options.add_argument("--disable-application-cache")
            chrome_options.add_argument("--disable-cache")
            chrome_options.add_argument("--disk-cache-size=0")
            chrome_options.add_argument("--media-cache-size=0")
            
            # 성능 관련 설정
            chrome_options.add_argument("--disable-background-networking")
            chrome_options.add_argument("--disable-background-timer-throttling")
            chrome_options.add_argument("--disable-backgrounding-occluded-windows")
            chrome_options.add_argument("--disable-breakpad")
            chrome_options.add_argument("--disable-component-extensions-with-background-pages")
            chrome_options.add_argument("--disable-default-apps")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-features=TranslateUI")
            chrome_options.add_argument("--disable-ipc-flooding-protection")
            chrome_options.add_argument("--disable-renderer-backgrounding")
            chrome_options.add_argument("--enable-features=NetworkService,NetworkServiceInProcess")
            chrome_options.add_argument("--force-color-profile=srgb")
            chrome_options.add_argument("--metrics-recording-only")
            chrome_options.add_argument("--no-first-run")
            chrome_options.add_argument("--password-store=basic")
            chrome_options.add_argument("--use-mock-keychain")
            
            # 로깅 비활성화
            chrome_options.add_argument("--log-level=3")
            chrome_options.add_argument("--silent")
            
            # Service 객체를 명시적으로 사용
            try:
                # ChromeDriverManager 초기화 방식 변경
                chrome_path = ChromeDriverManager().install()
                if "THIRD_PARTY_NOTICES.chromedriver" in chrome_path:
                    chrome_path = chrome_path.replace("THIRD_PARTY_NOTICES.chromedriver", "chromedriver.exe")
                service = Service(executable_path=chrome_path)
                
                # WebDriver 초기화 시도
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.driver = webdriver.Chrome(service=service, options=chrome_options)
                        self.driver.set_page_load_timeout(self.timeout)
                        self.logger.info("Selenium WebDriver initialized successfully")
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        self.logger.warning(f"WebDriver initialization attempt {attempt + 1} failed: {str(e)}")
                        time.sleep(2)  # 재시도 전 대기
                        
            except Exception as install_error:
                self.logger.error(f"Failed to install/find ChromeDriver via webdriver-manager: {install_error}")
                # 대체 방법 시도
                try:
                    from selenium.webdriver.chrome.service import Service as ChromeService
                    service = ChromeService()
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback ChromeDriver initialization failed: {fallback_error}")
                    raise

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

    def search_product(
        self, query: str, max_items: int = 50, keyword2: str = ""
    ) -> List[Product]:
        """고려기프트 상품 검색 (Selenium 전용)"""
        self.logger.info(f"Searching Koryo Gift (Selenium) for: {query}")

        # Directly use Selenium search logic
        try:
            results = self._search_product_logic(query, max_items=max_items)
            self.logger.info(f"Found {len(results)} products using Selenium.")
            return results
        except Exception as e:
            self.logger.error(
                f"Error during Selenium search for '{query}': {e}", exc_info=True
            )
            return [] # Return empty list on error

    def _search_product_logic(self, query: str, max_items: int = 50) -> List[Product]:
        products = []
        max_retries = 3
        retry_delay = 5  # seconds

        try:
            # 직접 search_url를 사용하여 검색 결과 페이지로 이동
            search_url = f"{self.search_url}?keyword={urllib.parse.quote(query)}"
            self.logger.debug(f"검색 URL: {search_url}")

            for attempt in range(max_retries):
                try:
                    self.driver.get(search_url)
                    sleep(3)  # 검색 결과 로딩 대기
                    self.logger.info(f"Search results page loaded: {self.driver.current_url}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.logger.error(f"Failed to load search page after {max_retries} attempts: {str(e)}")
                        return []
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    sleep(retry_delay)
                    # WebDriver 재초기화 시도
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.setup_selenium()

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

                        # 다음 페이지 로딩 시도
                        for attempt in range(max_retries):
                            try:
                                self.driver.get(next_url)
                                sleep(2)
                                page += 1
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    self.logger.error(f"Failed to load next page after {max_retries} attempts: {str(e)}")
                                    return products
                                self.logger.warning(f"Attempt {attempt + 1} failed to load next page: {str(e)}")
                                sleep(retry_delay)

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
        """제품 목록 항목에서 기본 정보 추출 (Null 체크 강화)"""
        try:
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

            if not title_element:
                self.logger.warning("Product title element not found in item")
                return {}
            title = title_element.text.strip()
            product_data["title"] = title

            # 링크 추출
            link_element = title_element # Usually the link is on the title element
            if not link_element.get('href'): # Check if title element itself has href
                link_element = element.select_one('a') # Fallback to first link in element

            if not link_element or not link_element.get('href'):
                self.logger.warning("Product link not found in item")
                return {}
            link = link_element.get('href')
            if not link.startswith("http"):
                if link.startswith("/"):
                    link = f"{self.base_url}{link}"
                else:
                    link = f"{self.base_url}/ez/{link}"
            product_data["link"] = link

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

            # 고유 ID 생성 (link가 None이 아님을 보장)
            product_id = hashlib.md5(link.encode()).hexdigest()
            product_data["product_id"] = product_id

            return product_data

        except Exception as e:
            self.logger.warning(f"Error extracting list item: {str(e)}")
            return {}

    def _get_product_details(self, item: Dict) -> Optional[Product]:
        """Selenium을 사용하여 제품 상세 정보 가져오기 (오류 처리 강화)"""
        if not item or not item.get("link") or not item.get("product_id"):
            self.logger.warning("Invalid item data passed to _get_product_details")
            return None

        cache_key = f"koryo_detail|{item['product_id']}"
        if self.cache:
            cached_result = self.get_sparse_data(cache_key)
            if cached_result:
                return cached_result

        url = item["link"]
        current_url = None # Initialize current_url

        try:
            # 현재 페이지 URL 저장 (상세 페이지 이동 전)
            current_url = self.driver.current_url

            # 상세 페이지 접속
            try:
                self.driver.get(url)
                # Wait for a specific element that indicates page load, e.g., product title or price
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".product_name, h3.prd_title, .view_title, #main_price, .price_num")) # Check multiple possible elements
                )
                self.logger.debug(f"Detail page loaded successfully: {url}")
            except TimeoutException:
                self.logger.warning(
                    f"Detail page load timed out for {url}"
                )
                self.driver.execute_script("window.stop();") # Stop loading
                # Even if it timed out, try to parse what we have
            except Exception as load_err:
                self.logger.error(f"Error loading detail page {url}: {load_err}")
                # If loading fails, try to navigate back immediately
                if current_url:
                    try:
                        self.driver.get(current_url)
                        sleep(1)
                    except Exception as back_err:
                        self.logger.error(f"Failed to navigate back after load error: {back_err}")
                return None

            # HTML 파싱
            soup = BeautifulSoup(self.driver.page_source, "lxml")

            # 상세 정보 추출 (Null 체크 강화)
            title = item.get("title", "")
            for selector in self.title_selectors:
                detail_title_element = soup.select_one(selector)
                if detail_title_element:
                    title = detail_title_element.text.strip()
                    break

            price = item.get("price", 0)
            for selector in self.price_selectors:
                detail_price_element = soup.select_one(selector)
                if detail_price_element:
                    price_text = detail_price_element.text.strip()
                    price_match = self.patterns["price_number"].search(price_text)
                    if price_match:
                        price = int(price_match.group().replace(",", ""))
                    break

            product_code = ""
            for selector in self.code_selectors:
                code_element = soup.select_one(selector)
                if code_element:
                    code_text = code_element.text.strip()
                    # ... (existing code extraction logic) ...
                    break # Found code, no need to check other selectors

            # ... (extract image_gallery, quantity_prices, specifications, description with null checks as needed) ...
            # Example for image gallery:
            image_gallery = []
            for selector in self.image_selectors:
                elements = soup.select(selector)
                if elements:
                    image_elements = elements
                    break # Found image elements
                else:
                    image_elements = [] # Ensure it's a list even if not found

            if image_elements:
                 for img in image_elements:
                     if img:
                         img_url = img.get("src", "")
                         if img_url:
                              # ... (make url absolute) ...
                              image_gallery.append(img_url)

            # ... Extract quantity_prices, specifications, description similarly ...

            # 제품 객체 생성
            product = Product(
                id=item.get("product_id"), # Already checked this exists
                name=title,
                price=price,
                source="koryo",
                original_input_data=item,
                url=url, # Use the navigated URL
                image_url=item.get("image", "") or (image_gallery[0] if image_gallery else ""),
                image_gallery=image_gallery,
                product_code=product_code,
                # Add other extracted fields like description, specs, quantity_prices
                fetched_at=datetime.now().isoformat(), # Add timestamp
            )
            # ... (set other product attributes) ...

            # 캐시에 저장
            if self.cache:
                 self.add_sparse_data(cache_key, product)  # 기본 캐시 기간 사용

            # 이전 페이지로 돌아가기 (current_url이 설정되었는지 확인)
            if current_url:
                self.driver.get(current_url)
                sleep(1)
            else:
                self.logger.warning("Could not navigate back, original URL was not stored.")

            return product

        except Exception as e:
            self.logger.error(
                f"Error getting product details for {url}: {str(e)}", exc_info=True
            )
            # 예외 발생 시 이전 페이지로 돌아가기 (current_url 확인)
            if current_url:
                try:
                    self.driver.get(current_url)
                    sleep(1)
                except Exception as back_e:
                    self.logger.error(f"Failed to navigate back after error: {back_e}")
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

    def _extract_product_data(
        self, soup: BeautifulSoup, product_idx: str, url: str
    ) -> Dict[str, Any]:
        """상품 데이터 추출"""
        product_data = {
            "source": "koryo",
            "product_id": product_idx,
            "title": "",
            "price": 0,
            "main_image": "",
            "image_gallery": [],
            "is_sold_out": False,
            "quantity_prices": {},
        }

        # 이미지 URL들을 저장할 리스트
        image_gallery = []

        try:
            # 1. 메인 이미지 추출
            main_img = soup.find('img', id='main_img')
            if main_img and main_img.get('src'):
                main_img_url = urljoin(url, main_img['src'])
                if main_img_url not in image_gallery:
                    image_gallery.append(main_img_url)
                    product_data["main_image"] = main_img_url

            # 2. 썸네일 이미지들 추출
            thumbnails = soup.select('.product_picture .thumnails img')
            for thumb in thumbnails:
                if thumb.get('src'):
                    thumb_url = urljoin(url, thumb['src'])
                    if thumb_url not in image_gallery:
                        image_gallery.append(thumb_url)

            # 3. 상세 이미지들 (상품 정보 테이블 내 이미지)
            detail_images = soup.select('.tbl_info img')
            for img in detail_images:
                if img.get('src'):
                    img_url = img['src']
                    if not img_url.startswith(('http://', 'https://')):
                        img_url = urljoin(url, img_url)
                    if img_url not in image_gallery:
                        image_gallery.append(img_url)

            # 4. 상세 이미지들 (상품 설명 영역의 모든 테이블)
            tables = soup.find_all('table')
            for table in tables:
                images = table.find_all('img')
                for img in images:
                    if img.get('src'):
                        img_url = img['src']
                        if not img_url.startswith(('http://', 'https://')):
                            img_url = urljoin(url, img_url)
                        if img_url not in image_gallery:
                            image_gallery.append(img_url)

            # 5. 리뷰 이미지들
            review_images = soup.select('.tbl_review img')
            for img in review_images:
                if img.get('src'):
                    img_url = urljoin(url, img['src'])
                    if img_url not in image_gallery:
                        image_gallery.append(img_url)

            # 6. 외부 도메인 이미지들 (상세 설명용)
            external_images = soup.find_all('img', src=re.compile(r'ain8949\.godohosting\.com'))
            for img in external_images:
                if img.get('src'):
                    img_url = img['src']
                    if img_url not in image_gallery:
                        image_gallery.append(img_url)

            # 7. UI 요소 이미지 필터링
            filtered_gallery = [
                img for img in image_gallery
                if not any(ui_pattern in img.lower() for ui_pattern in [
                    'btn_', 'icon_', 'dot_', 'bottom_', '/bbs/image/',
                    'guide1.jpg', 's1.png'
                ])
            ]

            # 이미지 갤러리 업데이트
            product_data["image_gallery"] = filtered_gallery

            # 메인 이미지가 없는 경우 첫 번째 이미지를 메인 이미지로 설정
            if not product_data["main_image"] and filtered_gallery:
                product_data["main_image"] = filtered_gallery[0]

            self.logger.debug(f"Found {len(filtered_gallery)} images for product {product_idx}")

        except Exception as e:
            self.logger.error(f"Error extracting images for product {product_idx}: {str(e)}")

        # ... (나머지 기존 코드는 유지) ...

        return product_data
