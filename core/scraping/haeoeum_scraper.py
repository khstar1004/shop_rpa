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
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Playwright 지원 (선택적 의존성)
try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightTimeoutError = None

from ..data_models import Product
from . import BaseMultiLayerScraper


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
        super().__init__(max_retries, cache, (connect_timeout, read_timeout), use_proxies, debug)
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
        self.session = self._create_robust_session()

    def _create_robust_session(self) -> requests.Session:
        """재시도 로직이 포함된 requests 세션을 생성합니다."""
        session = requests.Session()
        session.headers.update({'User-Agent': self.user_agent})
        session.verify = False

        # Retry 전략 설정
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)

        session.mount('http://', adapter)
        session.mount('https://', adapter)

        self.logger.info("Robust requests session created with retry strategy.")
        return session

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

        try:
            # --- 웹 페이지 요청 (강화된 세션 사용) ---
            response = self.session.get(
                url,
                timeout=self.timeout_config,
                headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            response.raise_for_status()
            self.logger.info(f"Successfully fetched HTML content for {url} (Status: {response.status_code})")

            # HTML 파싱
            try:
                soup = BeautifulSoup(response.content, "html.parser")
                self.logger.debug(f"HTML parsed successfully for {url}")
            except Exception as parse_err:
                self.logger.error(f"Failed to parse HTML for {url}: {parse_err}")
                if self.debug:
                    self.logger.error(f"Raw HTML content: {response.text[:500]}...")
                return None

            # --- 상품 정보 추출 ---
            try:
                product_data = self._extract_product_data(soup, product_idx, url)
                if not product_data:
                    self.logger.warning(f"Could not extract significant product data from {url}")
                    return None
                self.logger.info(f"Product data extracted successfully for p_idx={product_idx}")

            except Exception as extract_err:
                self.logger.error(f"Error during data extraction for {url}: {extract_err}", exc_info=self.debug)
                return None
            
            # --- Product 객체 생성 ---
            try:
                product = Product(
                    id=product_data.get("product_id", f"haeoreum_{product_idx}"),
                    name=product_data.get("title", ""),
                    source="haeoreum",
                    price=float(product_data.get("price", 0.0)),
                    url=url,
                    image_url=product_data.get("main_image", ""),
                    product_code=product_data.get("product_code", ""),
                    image_gallery=product_data.get("image_gallery", []),
                )

                # 수량별 가격 정보 추가
                if "quantity_prices" in product_data:
                    product.quantity_prices = product_data["quantity_prices"]

                # 품절 정보 추가
                product.is_in_stock = not product_data.get("is_sold_out", False)

                # 추가 정보 추가
                if "specifications" in product_data:
                    product.specifications = product_data["specifications"]

                if "description" in product_data:
                    product.description = product_data["description"]

                self.logger.debug(f"Product object created for p_idx={product_idx}")

                # 캐시에 저장
                self.add_sparse_data(cache_key, product, ttl=86400)
                self.logger.info(f"Product data for p_idx={product_idx} cached.")
                return product

            except (ValueError, TypeError) as create_err:
                self.logger.error(f"Error creating Product object for {url}: {create_err}", exc_info=self.debug)
                return None

        except requests.exceptions.Timeout as timeout_err:
            self.logger.error(f"Request timed out for {url} after {self.timeout_config}: {timeout_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            self.logger.error(f"Request failed for {url} after {self.max_retries} retries: {req_err}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred processing {url}: {e}", exc_info=self.debug)
            return None

    def _extract_product_data(
        self, soup: BeautifulSoup, product_idx: str, url: str
    ) -> Dict[str, Any]:
        """
        BeautifulSoup 객체에서 상품 정보를 추출합니다.
        """
        product_data = {}

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

        # 메인 이미지 URL 추출 - 중요: ID가 target_img인 이미지 찾기
        main_image_element = soup.select_one(self.selectors["main_image"]["selector"])
        if main_image_element:
            main_image = main_image_element.get("src", "")
        else:
            # 대체 셀렉터 시도
            alt_image_element = soup.select_one(
                self.selectors["alt_main_image"]["selector"]
            )
            main_image = alt_image_element.get("src", "") if alt_image_element else ""

        # URL 정규화
        if main_image and not main_image.startswith(("http://", "https://")):
            main_image = urljoin(self.BASE_URL, main_image)

        product_data["main_image"] = main_image

        # 모든 이미지 URL 추출 (메인 + 썸네일 + 설명 이미지)
        image_gallery = []

        # 메인 이미지 추가
        if main_image:
            image_gallery.append(main_image)

        # 썸네일 이미지 추가
        thumbnail_elements = soup.select(self.selectors["thumbnail_images"]["selector"])
        for img in thumbnail_elements:
            img_url = img.get("src", "")
            if img_url and not img_url.startswith(("http://", "https://")):
                img_url = urljoin(self.BASE_URL, img_url)

            # 중복 제거하고 추가
            if img_url and img_url not in image_gallery:
                image_gallery.append(img_url)

        # 설명 이미지 추가
        desc_img_elements = soup.select(
            self.selectors["description_images"]["selector"]
        )
        for img in desc_img_elements:
            img_url = img.get("src", "")
            if img_url and not img_url.startswith(("http://", "https://")):
                img_url = urljoin(self.BASE_URL, img_url)

            # 중복 제거하고 추가
            if img_url and img_url not in image_gallery:
                image_gallery.append(img_url)

        # 페이지 내 모든 img 태그 검색하여 놓친 이미지 없는지 확인
        all_images = soup.find_all("img")
        for img in all_images:
            img_url = img.get("src", "")
            # 필터링: 실제 상품 이미지만 추가, 아이콘 등은 제외
            if img_url and ("/upload/" in img_url or "/product/" in img_url):
                if img_url and not img_url.startswith(("http://", "https://")):
                    img_url = urljoin(self.BASE_URL, img_url)
                # 중복 제거하고 추가
                if img_url and img_url not in image_gallery:
                    image_gallery.append(img_url)

        product_data["image_gallery"] = image_gallery

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

        # 수량별 가격 추출
        quantity_prices = {}
        price_table = soup.select_one(self.selectors["price_table"]["selector"])
        if price_table:
            rows = price_table.select("tr")
            if len(rows) >= 2:
                quantity_cells = rows[0].select("td")[1:]  # 첫번째 셀(수량)은 제외
                price_cells = rows[1].select("td")[1:]  # 첫번째 셀(단가)은 제외

                for i in range(min(len(quantity_cells), len(price_cells))):
                    qty_text = quantity_cells[i].text.strip()
                    price_text = price_cells[i].text.strip()

                    # 숫자만 추출
                    qty_match = re.search(r"\d+", qty_text)
                    price_match = self.patterns["price_number"].search(price_text)

                    if qty_match and price_match:
                        qty = int(qty_match.group())
                        qty_price = int(price_match.group().replace(",", ""))
                        quantity_prices[str(qty)] = qty_price

        # 수량 드롭다운에서 정보 추출 시도
        if not quantity_prices:
            quantity_dropdown = soup.select_one(
                self.selectors["quantity_dropdown"]["selector"]
            )
            if quantity_dropdown:
                options = quantity_dropdown.select("option")
                for option in options:
                    option_text = option.text.strip()

                    # 텍스트에서 수량과 가격 추출 시도 (예: "100개 (+1,000원)")
                    qty_match = self.patterns["quantity"].search(option_text)
                    price_match = self.patterns["price_number"].search(option_text)

                    if qty_match and price_match:
                        qty = int(qty_match.group(1))
                        option_price = int(price_match.group().replace(",", ""))

                        # 추가 금액이면 기본 가격에 더함
                        if "+" in option_text:
                            option_price = price + option_price

                        quantity_prices[str(qty)] = option_price

        # 텍스트에서 수량별 가격 정보 찾기
        if not quantity_prices:
            text_content = soup.get_text()
            matches = self.patterns["quantity_price"].findall(text_content)

            if matches:
                for match in matches:
                    qty = int(match[0])
                    qty_price = int(match[1].replace(",", ""))

                    # VAT 포함 여부 확인
                    vat_included = bool(
                        self.patterns["vat_included"].search(text_content)
                    )
                    if not vat_included:
                        qty_price = int(qty_price * 1.1)  # VAT 10% 추가

                    quantity_prices[str(qty)] = qty_price

        product_data["quantity_prices"] = quantity_prices

        # 가격표가 있으면 CSV로 저장
        if quantity_prices:
            try:
                # 데이터프레임 생성
                df = pd.DataFrame(
                    {
                        "수량": [int(qty) for qty in quantity_prices.keys()],
                        "일반": [int(price) for price in quantity_prices.values()],
                    }
                )
                df = df.sort_values(by="수량")

                # CSV 저장
                output_file = (
                    f"{self.output_dir}/haeoeum_{product_id[:8]}_price_table.csv"
                )
                df.to_csv(output_file, index=False)
                self.logger.info(f"Price table saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving price table: {e}")

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
        """
        해오름 기프트에서 제품 검색

        Args:
            query: 검색어
            max_items: 최대 검색 결과 수

        Returns:
            List[Product]: 검색된 제품 목록
        """
        self.logger.error(
            "해오름 기프트는 검색 API를 제공하지 않습니다. ID로 직접 조회해주세요."
        )

        # 매뉴얼 요구사항: 찾지 못하면 "동일상품 없음"으로 처리
        no_match_product = Product(
            id="no_match_haeoeum",
            name=f"동일상품 없음 - {query}",
            source="haeoreum",
            price=0,
            url="",
            image_url="",
        )
        return [no_match_product]
