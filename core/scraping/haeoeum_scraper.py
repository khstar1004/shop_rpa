"""
Haeoeum Gift 웹사이트 스크래퍼 모듈

Haeoeum Gift(JCL Gift) 웹사이트에서 상품 정보를 추출하는 기능을 제공합니다.
특히 product_view.asp 페이지에서 이미지 및 상품 정보를 추출하는 기능에 최적화되어 있습니다.
"""

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Disable InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Playwright 지원 (선택적 의존성)
try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightTimeoutError = None

from ..data_models import Product
from .base_multi_layer_scraper import BaseMultiLayerScraper
from .utils import extract_main_image


class HaeoeumScraper(BaseMultiLayerScraper):
    """
    해오름 기프트(JCL Gift) 웹사이트 스크래퍼
    특히 이미지 URL이 제대로 추출되도록 최적화되어 있습니다.

    추가 기능:
    - 수량별 가격표 추출
    - 품절 상태 확인
    - Playwright 지원 (설치된 경우)
    """

    BASE_URL = "http://www.jclgift.com"
    PRODUCT_VIEW_URL = "http://www.jclgift.com/product/product_view.asp"

    def __init__(
        self,
        max_retries: int = 5,
        cache: Optional[Any] = None,
        timeout: int = 30,
        connect_timeout: int = 10,
        read_timeout: int = 20,
        backoff_factor: float = 0.5,
        use_proxies: bool = False,
        debug: bool = False,
        output_dir: str = "output",
        headless: bool = True,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    ):
        """스크래퍼 초기화"""
        super().__init__(max_retries=max_retries, timeout=timeout, cache=cache)

        # --- Base Attributes ---
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.output_dir = output_dir
        self.headless = headless
        self.user_agent = user_agent
        self.timeout_config = (connect_timeout, read_timeout)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # Playwright 사용 가능 여부 체크
        self.playwright_available = PLAYWRIGHT_AVAILABLE
        if not self.playwright_available:
            self.logger.warning(
                "Playwright is not installed. Some features will be limited."
            )

        # 대화 상자 메시지 저장용
        self.dialog_message = ""

        # 해오름 기프트 웹사이트 셀렉터 정의
        self.selectors = {
            # 상품명 관련 셀렉터
            "product_title": {
                "selector": 'td.Ltext[bgcolor="#F5F5F5"]',
                "options": {"multiple": False},
            },
            # 상품 코드 관련 셀렉터
            "product_code": {
                "selector": "td.code_b2 > button > b",
                "options": {"multiple": False},
            },
            # 메인 이미지 관련 셀렉터 (target_img)
            "main_image": {
                "selector": "img#target_img",
                "options": {"multiple": False, "attribute": "src"},
            },
            # 대체 메인 이미지 셀렉터
            "alt_main_image": {
                "selector": 'td[height="340"] img',
                "options": {"multiple": False, "attribute": "src"},
            },
            # 썸네일 이미지 셀렉터
            "thumbnail_images": {
                "selector": 'table[width="62"] img',
                "options": {"multiple": True, "attribute": "src"},
            },
            # 가격 관련 셀렉터들
            "applied_price": {"selector": "#price_e", "options": {"multiple": False}},
            "total_price": {"selector": "#buy_price", "options": {"multiple": False}},
            # 가격표 셀렉터
            "price_table": {"selector": "table.pv_tbl", "options": {"multiple": False}},
            # 상품 상세 정보 테이블
            "product_info_table": {
                "selector": "table.tbl_contab",
                "options": {"multiple": False},
            },
            # 상품 설명 이미지
            "description_images": {
                "selector": "div.product_view_img img",
                "options": {"multiple": True, "attribute": "src"},
            },
            # 수량 선택 드롭다운
            "quantity_dropdown": {
                "selector": 'select[name="goods_ea"], select[name="item_ea"], select.ea_sel',
                "options": {"multiple": False},
            },
            # 품절 표시
            "sold_out": {
                "selector": 'span.soldout, div.soldout, img[src*="soldout"], div:contains("품절")',
                "options": {"multiple": False},
            },
        }

        # 텍스트 추출용 정규식 패턴
        self.patterns = {
            "price_number": re.compile(r"[\d,]+"),
            "product_code": re.compile(r"상품코드\s*:\s*([A-Za-z0-9-]+)"),
            "quantity": re.compile(r"(\d+)(개|세트|묶음)"),
            "quantity_price": re.compile(
                r"(\d+)개[:\s]+([0-9,]+)원"
            ),  # 수량:가격 패턴 (예: "100개: 1,000원")
            "vat_included": re.compile(r"VAT\s*(포함|별도|제외)", re.IGNORECASE),
        }

        # --- requests 세션 설정 강화 ---
        self.session = requests.Session()

        # 프록시 설정 (필요한 경우)
        # ... (proxy logic)

        # 재시도 전략 및 어댑터 설정
        try:
            retry_strategy = Retry(
                total=max_retries, # Use max_retries from init
                backoff_factor=self.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"], # Ensure this is allowed_methods
            )
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            self.logger.info(f"Retry strategy configured with max_retries={max_retries}")
        except Exception as e:
            self.logger.error(f"Failed to configure retry strategy: {e}")

        # 헤더 설정
        self.session.headers.update({
            "User-Agent": self.user_agent
        })

    def get_product(self, product_idx: str) -> Optional[Product]:
        """
        해오름 기프트 상품 페이지에서 상품 정보를 추출합니다.

        Args:
            product_idx: 상품 ID (p_idx 파라미터 값)

        Returns:
            Product 객체 또는 None (실패 시)
        """
        cache_key = f"haeoeum_product|{product_idx}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            self.logger.info(f"Using cached product data for p_idx={product_idx}")
            return cached_result

        url = f"{self.PRODUCT_VIEW_URL}?p_idx={product_idx}"
        self.logger.info(f"Attempting to fetch product data from: {url}")

        # Playwright가 사용 가능한 경우 먼저 시도
        if self.playwright_available:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self.headless)
                    page = browser.new_page()
                    page.goto(url, wait_until="networkidle")
                    
                    # 이미지가 로드될 때까지 대기
                    page.wait_for_selector("img", timeout=5000)
                    
                    # 페이지의 HTML 가져오기
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # 이미지 URL 추출
                    images = page.evaluate("""() => {
                        const images = [];
                        document.querySelectorAll('img').forEach(img => {
                            if (img.src && (img.src.includes('/upload/') || img.src.includes('/product/'))) {
                                images.push(img.src);
                            }
                        });
                        return images;
                    }""")
                    
                    if images:
                        self.logger.info(f"Found {len(images)} images using Playwright")
                        product_data = self._extract_product_data(soup, product_idx, url)
                        product_data["image_gallery"] = images
                        product = Product(**product_data)
                        self.cache_sparse_data(cache_key, product)
                        return product
                        
            except Exception as e:
                self.logger.warning(f"Playwright extraction failed: {e}")

        # 기존 requests 방식으로 시도
        try:
            response = self.session.get(url, timeout=self.timeout_config)
            response.raise_for_status()
            self.logger.info(f"Successfully fetched HTML content for {url} (Status: {response.status_code})")
            
            soup = BeautifulSoup(response.text, "html.parser")
            product_data = self._extract_product_data(soup, product_idx, url)
            
            if product_data:
                product = Product(**product_data)
                self.cache_sparse_data(cache_key, product)
                self.logger.info(f"Product data for p_idx={product_idx} cached.")
                return product
                
        except Exception as e:
            self.logger.error(f"Failed to fetch product data: {e}")
            
        return None

    def _extract_product_data(
        self, soup: BeautifulSoup, product_idx: str, url: str
    ) -> Dict[str, Any]:
        """상품 데이터 추출"""
        product_data = {
            "source": "haeoreum",
            "product_id": product_idx,
            "title": "",
            "price": 0,
            "main_image": "",
            "image_gallery": [],
            "is_sold_out": False,
            "quantity_prices": {},
        }

        # 모든 상품 이미지 URL 추출
        image_gallery = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and '/upload/product/' in src:
                # 작은 이미지(simg3)를 큰 이미지(bimg3)로 변경
                if 'simg3' in src:
                    src = src.replace('simg3', 'bimg3')
                full_url = urljoin(self.BASE_URL, src)
                if full_url not in image_gallery:  # 중복 제거
                    image_gallery.append(full_url)

        # 이미지가 있으면 첫 번째 이미지를 메인 이미지로 설정
        if image_gallery:
            product_data["main_image"] = image_gallery[0]
            product_data["image_gallery"] = image_gallery

        # 상품명 추출
        title_element = soup.select_one(self.selectors["product_title"]["selector"])
        title = title_element.text.strip() if title_element else ""

        if not title:
            # 타이틀 태그에서 추출 시도
            title = soup.title.string.strip() if soup.title else ""
            if ">" in title:
                # 타이틀에서 불필요한 사이트명 제거
                title = title.split(">", 1)[0].strip()

        product_data["title"] = title

        # 상품 코드 추출
        code_element = soup.select_one(self.selectors["product_code"]["selector"])
        product_code = code_element.text.strip() if code_element else ""

        # 코드가 없으면 URL에서 추출 시도
        if not product_code:
            product_code = product_idx

        product_data["product_code"] = product_code

        # 고유 ID 생성
        product_id = hashlib.md5(f"haeoeum_{product_code}".encode()).hexdigest()
        product_data["product_id"] = product_id

        # 가격 추출
        price_element = soup.select_one(self.selectors["applied_price"]["selector"])
        price_text = price_element.text.strip() if price_element else "0"
        price_match = self.patterns["price_number"].search(price_text)
        price = int(price_match.group().replace(",", "")) if price_match else 0

        # 가격이 0이면 총합계금액에서 시도
        if price == 0:
            total_price_element = soup.select_one(
                self.selectors["total_price"]["selector"]
            )
            if total_price_element:
                price_text = total_price_element.text.strip()
                price_match = self.patterns["price_number"].search(price_text)
                price = int(price_match.group().replace(",", "")) if price_match else 0

        product_data["price"] = price

        # 품절 여부 확인
        sold_out_element = soup.select_one(self.selectors["sold_out"]["selector"])
        is_sold_out = bool(sold_out_element)

        # 텍스트에서 품절 키워드 확인
        if not is_sold_out:
            page_text = soup.get_text().lower()
            if "품절" in page_text or "sold out" in page_text:
                is_sold_out = True

        product_data["is_sold_out"] = is_sold_out

        # 수량별 가격 추출 - 개선된 버전
        quantity_prices = {}
        price_table = soup.select_one(self.selectors["price_table"]["selector"])
        
        if price_table:
            # 테이블 헤더 찾기
            headers = []
            header_row = price_table.select_one("tr:first-child")
            if header_row:
                headers = [th.text.strip() for th in header_row.select("th, td")]
            
            # 데이터 행 처리
            data_rows = price_table.select("tr:not(:first-child)")
            for row in data_rows:
                cells = row.select("td")
                if len(cells) >= 2:
                    # 수량과 가격 추출
                    qty_text = cells[0].text.strip()
                    price_text = cells[1].text.strip()
                    
                    # 수량 추출 (숫자만)
                    qty_match = re.search(r"\d+", qty_text)
                    if qty_match:
                        qty = int(qty_match.group())
                        
                        # 가격 추출 (숫자만)
                        price_match = self.patterns["price_number"].search(price_text)
                        if price_match:
                            price = int(price_match.group().replace(",", ""))
                            
                            # VAT 포함 여부 확인
                            vat_included = bool(self.patterns["vat_included"].search(price_text))
                            if not vat_included:
                                price = int(price * 1.1)  # VAT 10% 추가
                                
                            quantity_prices[str(qty)] = price

        # 수량 드롭다운에서 정보 추출 시도
        if not quantity_prices:
            quantity_dropdown = soup.select_one(
                self.selectors["quantity_dropdown"]["selector"]
            )
            if quantity_dropdown:
                options = quantity_dropdown.select("option")
                for option in options:
                    option_text = option.text.strip()
                    
                    # 수량과 가격 추출
                    qty_match = self.patterns["quantity"].search(option_text)
                    price_match = self.patterns["price_number"].search(option_text)
                    
                    if qty_match and price_match:
                        qty = int(qty_match.group(1))
                        price = int(price_match.group().replace(",", ""))
                        
                        # 추가 금액이면 기본 가격에 더함
                        if "+" in option_text:
                            price = product_data["price"] + price
                            
                        # VAT 포함 여부 확인
                        vat_included = bool(self.patterns["vat_included"].search(option_text))
                        if not vat_included:
                            price = int(price * 1.1)  # VAT 10% 추가
                            
                        quantity_prices[str(qty)] = price

        # 텍스트에서 수량별 가격 정보 찾기
        if not quantity_prices:
            text_content = soup.get_text()
            matches = self.patterns["quantity_price"].findall(text_content)
            
            if matches:
                for match in matches:
                    qty = int(match[0])
                    price = int(match[1].replace(",", ""))
                    
                    # VAT 포함 여부 확인
                    vat_included = bool(self.patterns["vat_included"].search(text_content))
                    if not vat_included:
                        price = int(price * 1.1)  # VAT 10% 추가
                        
                    quantity_prices[str(qty)] = price

        product_data["quantity_prices"] = quantity_prices

        # 가격표가 있으면 CSV로 저장
        if quantity_prices:
            try:
                # 수량과 가격을 정렬
                sorted_quantities = sorted([int(qty) for qty in quantity_prices.keys()])
                sorted_prices = [quantity_prices[str(qty)] for qty in sorted_quantities]
                
                # 고려 가격 차이 계산
                koryo_price_differences = [0]  # 첫 번째 항목은 0
                koryo_price_difference_percentages = [0]  # 첫 번째 항목은 0
                
                for i in range(1, len(sorted_prices)):
                    diff = sorted_prices[i] - sorted_prices[i-1]
                    koryo_price_differences.append(diff)
                    
                    # 가격 차이 백분율 계산 (이전 가격 대비)
                    if sorted_prices[i-1] > 0:
                        percentage = round((diff / sorted_prices[i-1]) * 100, 2)
                    else:
                        percentage = 0
                    koryo_price_difference_percentages.append(percentage)

                # TODO: 네이버 가격 데이터 가져오는 로직 추가 필요
                naver_prices = [0] * len(sorted_quantities) # 임시로 0으로 초기화
                naver_base_quantity = [0] * len(sorted_quantities) # 임시로 0으로 초기화

                # 네이버 가격 차이 계산 (임시: 고려 가격 - 네이버 가격)
                naver_price_differences = []
                naver_price_difference_percentages = []
                for koryo_price, naver_price in zip(sorted_prices, naver_prices):
                    # 네이버 가격이 유효할 때만 계산 (현재는 항상 0)
                    if naver_price > 0:
                        naver_diff = koryo_price - naver_price
                        naver_percentage = round((naver_diff / naver_price) * 100, 2) if naver_price > 0 else 0
                    else:
                        naver_diff = 0
                        naver_percentage = 0
                    naver_price_differences.append(naver_diff)
                    naver_price_difference_percentages.append(naver_percentage)

                # 데이터프레임 생성 (네이버 컬럼 추가)
                df = pd.DataFrame({
                    "고려 기본수량": sorted_quantities,
                    "판매단가2(VAT포함)": sorted_prices, # 고려 가격
                    "고려 가격차이": koryo_price_differences,
                    "고려 가격차이(%)": koryo_price_difference_percentages,
                    "고려 링크": [url for _ in sorted_quantities],
                    "네이버 기본수량": naver_base_quantity, # 실제 네이버 데이터로 채워야 함
                    "판매단가3(VAT포함)": naver_prices, # 실제 네이버 데이터로 채워야 함 (컬럼명 확인 필요)
                    "네이버 가격차이": naver_price_differences, # 계산된 네이버 가격 차이
                    "네이버 가격차이(%)": naver_price_difference_percentages # 계산된 네이버 가격 차이(%)
                })
                
                # CSV 저장
                output_file = f"{self.output_dir}/haeoeum_{product_id[:8]}_price_table.csv"
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                self.logger.info(f"가격표 저장 완료: {output_file}")
                
                # 데이터 로깅
                self.logger.info(f"추출된 고려 수량: {sorted_quantities}")
                self.logger.info(f"추출된 고려 가격: {sorted_prices}")
                self.logger.info(f"계산된 고려 가격 차이: {koryo_price_differences}")
                self.logger.info(f"계산된 고려 가격 차이(%): {koryo_price_difference_percentages}")
                # self.logger.info(f"추출된 네이버 가격: {naver_prices}") # 네이버 가격 로깅 (추후 추가)
                # self.logger.info(f"계산된 네이버 가격 차이: {naver_price_differences}") # 네이버 가격 차이 로깅
                # self.logger.info(f"계산된 네이버 가격 차이(%): {naver_price_difference_percentages}") # 네이버 가격 차이(%) 로깅
                
            except Exception as e:
                self.logger.error(f"가격표 저장 오류: {e}")

        # 사양 정보 추출
        specifications = {}
        specs_table = soup.select_one(self.selectors["product_info_table"]["selector"])
        if specs_table:
            rows = specs_table.select("tr")
            for row in rows:
                th = row.select_one("th")
                td = row.select_one("td")
                if th and td:
                    key = th.text.strip().replace("·", "").strip()
                    value = td.text.strip()
                    if key and value:
                        specifications[key] = value

        product_data["specifications"] = specifications

        # 제품 설명 추출
        description = ""
        desc_elements = soup.select(
            "div.product_view_img, div.prd_detail, div.item_detail"
        )
        if desc_elements:
            description = "\n".join([elem.text.strip() for elem in desc_elements])

        product_data["description"] = description

        return product_data

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
                sold_out_element = page.locator(
                    'span.soldout, div.soldout, img[src*="soldout"], div:has-text("품절")'
                ).first
                if sold_out_element and sold_out_element.count() > 0:
                    is_sold_out = True
                    sold_out_text = sold_out_element.text_content() or "품절"
                    self.logger.info(f"상품 품절 확인: {sold_out_text}")
                    browser.close()
                    return None, is_sold_out, sold_out_text

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
                    "table.pv_tbl, table.price_table"
                ).first
                if price_table_element and price_table_element.count() > 0:
                    html_content = price_table_element.inner_html()
                    soup = BeautifulSoup(html_content, "html.parser")

                    try:
                        # HTML에서 테이블 추출
                        table_df = pd.read_html(str(soup))[0]

                        # 수량 열과 가격 열 찾기
                        qty_col = None
                        price_col = None

                        for col in table_df.columns:
                            col_str = str(col).lower()
                            if (
                                "수량" in col_str
                                or "qty" in col_str
                                or "개수" in col_str
                            ):
                                qty_col = col
                            elif (
                                "가격" in col_str
                                or "price" in col_str
                                or "단가" in col_str
                            ):
                                price_col = col

                        if qty_col is not None and price_col is not None:
                            # 열 이름으로 매핑
                            df = pd.DataFrame(
                                {"수량": table_df[qty_col], "일반": table_df[price_col]}
                            )
                        else:
                            # 첫 번째와 두 번째 열 사용
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

                        # VAT 여부 확인
                        html_text = page.content().lower()
                        vat_included = (
                            "vat 포함" in html_text or "vat included" in html_text
                        )
                        if not vat_included:
                            df["일반"] = df["일반"] * 1.1  # VAT 10% 추가

                        # 정렬
                        df = df.sort_values(by="수량")

                        # CSV 저장
                        output_file = f"{self.output_dir}/haeoeum_price_table.csv"
                        df.to_csv(output_file, index=False)
                        self.logger.info(f"가격표 저장 완료: {output_file}")

                        browser.close()
                        return df, False, ""
                    except Exception as e:
                        self.logger.error(f"가격표 추출 오류: {e}")

                # 수량 드롭다운 확인
                quantity_dropdown = page.locator(
                    'select[name="goods_ea"], select[name="item_ea"], select.ea_sel'
                ).first
                if quantity_dropdown and quantity_dropdown.count() > 0:
                    options = []
                    prices = []

                    # 옵션 텍스트 가져오기
                    option_elements = quantity_dropdown.locator("option")
                    option_count = option_elements.count()

                    for i in range(option_count):
                        option_text = option_elements.nth(i).text_content()

                        # 수량 추출
                        qty_match = re.search(r"(\d+)개", option_text)
                        if qty_match:
                            qty = int(qty_match.group(1))

                            # 가격 추출
                            price_match = re.search(r"\+\s*([\d,]+)원", option_text)
                            if price_match:
                                # 기본 가격 + 추가 가격
                                try:
                                    base_price_element = page.locator(
                                        "#price_e, .prd_price, .item_price"
                                    ).first
                                    base_price_text = (
                                        base_price_element.text_content()
                                        if base_price_element.count() > 0
                                        else "0"
                                    )
                                    base_price_match = re.search(
                                        r"([\d,]+)", base_price_text
                                    )
                                    base_price = (
                                        int(base_price_match.group(1).replace(",", ""))
                                        if base_price_match
                                        else 0
                                    )

                                    # 추가 가격
                                    add_price = int(
                                        price_match.group(1).replace(",", "")
                                    )

                                    # 총 가격
                                    total_price = base_price + add_price

                                    options.append(qty)
                                    prices.append(total_price)
                                except Exception as e:
                                    self.logger.error(
                                        f"Error parsing option price: {e}"
                                    )

                    if options and prices:
                        # 데이터프레임 생성
                        df = pd.DataFrame({"수량": options, "일반": prices})

                        # VAT 여부 확인
                        html_text = page.content().lower()
                        vat_included = (
                            "vat 포함" in html_text or "vat included" in html_text
                        )
                        if not vat_included:
                            df["일반"] = df["일반"] * 1.1  # VAT 10% 추가

                        # 정렬
                        df = df.sort_values(by="수량")

                        # CSV 저장
                        output_file = (
                            f"{self.output_dir}/haeoeum_price_table_dropdown.csv"
                        )
                        df.to_csv(output_file, index=False)
                        self.logger.info(
                            f"드롭다운에서 추출한 가격표 저장 완료: {output_file}"
                        )

                        browser.close()
                        return df, False, ""

                # 페이지 텍스트에서 수량별 가격 정보 찾기
                html_content = page.content()
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text()

                # 정규식으로 수량:가격 패턴 찾기
                matches = re.findall(r"(\d+)개[:\s]+([0-9,]+)원", text_content)

                if matches:
                    quantities = [int(match[0]) for match in matches]
                    prices = [int(match[1].replace(",", "")) for match in matches]

                    # VAT 여부 확인
                    vat_included = (
                        "vat 포함" in text_content.lower()
                        or "vat included" in text_content.lower()
                    )
                    if not vat_included:
                        prices = [int(price * 1.1) for price in prices]

                    # 데이터프레임 생성
                    df = pd.DataFrame({"수량": quantities, "일반": prices})
                    df = df.sort_values(by="수량")

                    # CSV 저장
                    output_file = f"{self.output_dir}/haeoeum_price_table_text.csv"
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

    def search_product(self, query: str, max_items: int = 50) -> List[Product]:
        """해오름 기프트에서 상품을 검색합니다."""
        self.logger.info(f"해오름 기프트에서 '{query}' 검색 시작")
        
        # 캐시 확인
        cache_key = f"haeoeum_search|{query}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 검색 결과 사용: '{query}'")
            return cached_result

        try:
            # 검색 URL 구성
            search_url = f"{self.BASE_URL}/product/product_list.asp"
            params = {
                "search_keyword": query,
                "search_type": "all"
            }
            
            response = self.session.get(search_url, params=params, verify=False)
            response.raise_for_status()
            
            soup = self._get_soup(response.text)
            product_elements = soup.select(self.selectors["product_list"]["selector"])
            
            if not product_elements:
                self.logger.warning(f"❌ 상품이 존재하지 않음: '{query}' 검색 결과 없음")
                return []
            
            products = []
            for element in product_elements[:max_items]:
                try:
                    product_link = element.select_one(self.selectors["product_link"]["selector"])
                    if not product_link:
                        continue
                    
                    product_url = urljoin(self.BASE_URL, product_link["href"])
                    product_idx = self._extract_product_idx(product_url)
                    
                    if not product_idx:
                        continue
                    
                    product = self.get_product(product_idx)
                    if product:
                        products.append(product)
                    
                except Exception as e:
                    self.logger.error(f"상품 추출 중 오류 발생: {str(e)}")
                    continue
            
            if not products:
                self.logger.warning(f"❌ 상품이 존재하지 않음: '{query}' 검색 결과 없음")
                return []
            
            self.logger.info(f"해오름 기프트에서 '{query}' 검색 완료 - {len(products)}개 상품 발견")
            return products
            
        except Exception as e:
            self.logger.error(f"해오름 기프트 검색 중 오류 발생: {str(e)}")
            return []
