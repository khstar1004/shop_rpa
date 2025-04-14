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
        max_retries: int = 3,  # 재시도 횟수 감소
        cache: Optional[Any] = None,
        timeout: int = 15,  # 타임아웃 감소
        connect_timeout: int = 5,  # 연결 타임아웃 감소
        read_timeout: int = 10,  # 읽기 타임아웃 감소
        backoff_factor: float = 0.3,  # 백오프 팩터 감소
        use_proxies: bool = False,
        debug: bool = False,
        output_dir: str = "output",
        headless: bool = True,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        cache_ttl: int = 3600,  # 캐시 TTL 추가 (1시간)
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
        self.cache_ttl = cache_ttl

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # Playwright 설정 최적화
        if PLAYWRIGHT_AVAILABLE:
            self.playwright_config = {
                "viewport": {"width": 1280, "height": 720},  # 뷰포트 크기 최적화
                "timeout": 15000,  # Playwright 타임아웃 15초로 설정
                "wait_until": "domcontentloaded"  # 더 빠른 페이지 로드 조건
            }
        else:
            self.playwright_available = False
            self.logger.warning("Playwright is not installed. Some features will be limited.")

        # requests 세션 최적화
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # 연결 풀 크기 최적화
            pool_maxsize=20,
            pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": user_agent})

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
            # 상품 목록 셀렉터 (search_product 메서드용)
            "product_list": {
                "selector": "ul.prdList li, div.prdList > div, table.product_list tr",
                "options": {"multiple": True},
            },
            # 상품 링크 셀렉터 (search_product 메서드용)
            "product_link": {
                "selector": "a.prd_name, a.product_name, td.prdImg a, div.thumbnail a",
                "options": {"multiple": False, "attribute": "href"},
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

    def get_cached_data(self, key: str) -> Optional[Any]:
        """TTL이 적용된 캐시 데이터 조회"""
        if not self.cache:
            return None
            
        data = self.cache.get(key)
        if not data:
            return None
            
        # TTL 확인
        timestamp = data.get('timestamp', 0)
        if time.time() - timestamp > self.cache_ttl:
            return None
            
        return data.get('value')

    def cache_sparse_data(self, key: str, value: Any) -> None:
        """TTL이 적용된 캐시 데이터 저장"""
        if not self.cache:
            return
            
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

    def get_product(self, product_idx: str) -> Optional[Product]:
        """상품 정보를 가져옵니다."""
        cache_key = f"haeoeum_product_{product_idx}"
        cached_data = self.get_cached_data(cache_key)
        if cached_data:
            return cached_data

        url = f"{self.PRODUCT_VIEW_URL}?p_idx={product_idx}"
        
        # 먼저 requests로 시도 (더 빠름)
        try:
            response = self.session.get(url, timeout=self.timeout_config)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            product_data = self._extract_product_data(soup, product_idx, url)
            
            if product_data and product_data["status"] == "OK":
                product = Product(**product_data)
                self.cache_sparse_data(cache_key, product)
                return product
                
        except Exception as e:
            self.logger.warning(f"Requests extraction failed: {e}")

        # requests 실패 또는 불완전한 데이터인 경우 Playwright 시도
        if self.playwright_available:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self.headless)
                    page = browser.new_page(**self.playwright_config)
                    page.on("dialog", self._handle_dialog)
                    
                    page.goto(url, wait_until=self.playwright_config["wait_until"], timeout=self.playwright_config["timeout"])
                    
                    html_content = page.content()
                    soup = BeautifulSoup(html_content, "html.parser")
                    product_data = self._extract_product_data(soup, product_idx, url)
                    
                    browser.close()
                    
                    if product_data:
                        product = Product(**product_data)
                        self.cache_sparse_data(cache_key, product)
                        return product
                    
            except Exception as e:
                self.logger.error(f"Playwright extraction failed: {e}")
            
        return None

    def _sanitize_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Product 데이터를 정리하고 유효한 필드만 유지합니다."""
        # product_id를 id로 변환
        if "product_id" in product_data:
            product_data["id"] = product_data.pop("product_id")
        
        # Product 클래스의 유효한 필드 목록
        valid_fields = {
            "id", "name", "source", "price", "url", "image_url", "status",
            "brand", "description", "is_promotional_site", "product_code",
            "category", "min_order_quantity", "total_price_ex_vat",
            "total_price_incl_vat", "shipping_info", "specifications",
            "quantity_prices", "options", "reviews", "image_gallery",
            "stock_status", "delivery_time", "customization_options",
            "original_input_data"
        }
        
        # 유효하지 않은 필드 제거
        return {k: v for k, v in product_data.items() if k in valid_fields}

    def _extract_product_data(
        self, soup: BeautifulSoup, product_idx: str, url: str
    ) -> Dict[str, Any]:
        """상품 데이터 추출"""
        product_data = {
            "source": "haeoreum",
            "id": product_idx,
            "name": "",
            "price": 0,
            "image_url": "",
            "image_gallery": [],
            "status": None,  # 상태 필드 기본값
        }

        # 이미지 추출 최적화 - 한 번의 순회로 모든 이미지 수집
        image_gallery = set()  # 중복 방지를 위해 set 사용
        
        # 1. 메인 이미지 우선 추출
        main_img = soup.select_one(self.selectors["main_image"]["selector"])
        if main_img and main_img.get('src'):
            main_url = urljoin(self.BASE_URL, main_img['src'])
            if self._is_valid_image_url(main_url):
                image_gallery.add(main_url)
                product_data["image_url"] = main_url

        # 2. 모든 이미지 한 번에 추출
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if not src:
                continue
                
            # 이미지 URL 정규화
            img_url = urljoin(self.BASE_URL, src)
            if self._is_valid_image_url(img_url):
                image_gallery.add(img_url)

        # 3. 백그라운드 이미지 추출 (style 속성)
        for element in soup.find_all(style=True):
            style = element['style']
            if 'url(' in style:
                matches = re.findall(r'url\([\'"]?([^\'"()]+)[\'"]?\)', style)
                for match in matches:
                    img_url = urljoin(self.BASE_URL, match)
                    if self._is_valid_image_url(img_url):
                        image_gallery.add(img_url)

        # 이미지 갤러리 설정
        product_data["image_gallery"] = list(image_gallery)
        
        if not product_data["image_url"] and image_gallery:
            product_data["image_url"] = next(iter(image_gallery))

        # 상품명 추출 최적화
        title_element = soup.select_one(self.selectors["product_title"]["selector"])
        product_data["name"] = title_element.text.strip() if title_element else ""

        if not product_data["name"]:
            title = soup.title.string.strip() if soup.title else ""
            if ">" in title:
                product_data["name"] = title.split(">", 1)[0].strip()

        # 상품 코드 추출
        code_element = soup.select_one(self.selectors["product_code"]["selector"])
        product_data["product_code"] = code_element.text.strip() if code_element else product_idx

        # 고유 ID 생성
        product_data["id"] = hashlib.md5(f"haeoeum_{product_data['product_code']}".encode()).hexdigest()

        # 가격 추출 최적화
        price = 0
        price_element = soup.select_one(self.selectors["applied_price"]["selector"])
        if price_element:
            price_text = price_element.text.strip()
            price_match = self.patterns["price_number"].search(price_text)
            if price_match:
                price = int(price_match.group().replace(",", ""))

        if price == 0:
            total_price_element = soup.select_one(self.selectors["total_price"]["selector"])
            if total_price_element:
                price_text = total_price_element.text.strip()
                price_match = self.patterns["price_number"].search(price_text)
                if price_match:
                    price = int(price_match.group().replace(",", ""))

        product_data["price"] = price

        # 수량별 가격 추출 최적화
        product_data["quantity_prices"] = self._extract_quantity_prices(soup)

        # 상품 URL 추가
        product_data["url"] = url

        # 상태 설정
        if not product_data["name"]:
            product_data["status"] = "Title Not Found"
        elif not product_data["price"]:
            product_data["status"] = "Price Not Found"
        elif not product_data["image_url"]:
            product_data["status"] = "Image Not Found"
        else:
            product_data["status"] = "OK"

        return product_data

    def _is_valid_image_url(self, url: str) -> bool:
        """이미지 URL이 유효한지 확인"""
        if not url:
            return False
            
        # 이미지 파일 확장자 확인
        if not re.search(r'\.(jpg|jpeg|png|gif)([\?#].*)?$', url.lower()):
            return False
            
        # 제외할 패턴
        exclude_patterns = ['btn_', 'icon_', 'pixel.gif', 'button', 'bg_']
        return not any(pattern in url.lower() for pattern in exclude_patterns)

    def _extract_quantity_prices(self, soup: BeautifulSoup) -> Dict[str, int]:
        """수량별 가격 추출"""
        quantity_prices = {}
        
        # 1. 가격표에서 추출
        price_table = soup.select_one(self.selectors["price_table"]["selector"])
        if price_table:
            rows = price_table.select("tr:not(:first-child)")
            for row in rows:
                cells = row.select("td")
                if len(cells) >= 2:
                    qty_match = re.search(r"\d+", cells[0].text.strip())
                    price_match = self.patterns["price_number"].search(cells[1].text.strip())
                    if qty_match and price_match:
                        qty = int(qty_match.group())
                        price = int(price_match.group().replace(",", ""))
                        quantity_prices[str(qty)] = price
                        
        # 2. 드롭다운에서 추출 (가격표가 없는 경우)
        if not quantity_prices:
            dropdown = soup.select_one(self.selectors["quantity_dropdown"]["selector"])
            if dropdown:
                for option in dropdown.select("option"):
                    option_text = option.text.strip()
                    qty_match = self.patterns["quantity"].search(option_text)
                    price_match = self.patterns["price_number"].search(option_text)
                    if qty_match and price_match:
                        qty = int(qty_match.group(1))
                        price = int(price_match.group().replace(",", ""))
                        quantity_prices[str(qty)] = price

        return quantity_prices

    def _handle_dialog(self, dialog):
        """대화 상자 처리 (품절 등 상태 메시지 확인용)"""
        self.dialog_message = dialog.message
        self.logger.debug(f"Dialog message: {dialog.message}")
        dialog.accept()

    def _get_soup(self, html_content: str) -> BeautifulSoup:
        """HTML 문자열에서 BeautifulSoup 객체 생성"""
        return BeautifulSoup(html_content, "html.parser")
        
    def _extract_product_idx(self, url: str) -> Optional[str]:
        """URL에서 제품 ID(p_idx) 추출"""
        try:
            parsed_url = urlparse(url)
            # Query string 파싱
            from urllib.parse import parse_qs
            query_params = parse_qs(parsed_url.query)
            
            # p_idx 파라미터 찾기
            if 'p_idx' in query_params:
                return query_params['p_idx'][0]
                
            # 만약 p_idx가 없다면 다른 가능한 ID 파라미터 확인
            for param in ['idx', 'id', 'no', 'product_id']:
                if param in query_params:
                    return query_params[param][0]
            
            return None
        except Exception as e:
            self.logger.error(f"URL에서 제품 ID 추출 실패: {e}")
            return None

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
                    soup = self._get_soup(html_content)

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
                soup = self._get_soup(html_content)
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

    def test_image_extraction(self, product_idx: str) -> Dict[str, Any]:
        """
        특정 상품의 이미지 추출 테스트를 수행합니다.
        이 메서드는 테스트 및 디버깅 목적으로 사용됩니다.
        
        Args:
            product_idx: 상품 ID (p_idx 파라미터 값)
            
        Returns:
            이미지 추출 결과를 포함하는 딕셔너리
        """
        url = f"{self.PRODUCT_VIEW_URL}?p_idx={product_idx}"
        self.logger.info(f"Testing image extraction for product: {url}")
        
        test_results = {
            "product_idx": product_idx,
            "url": url,
            "requests_method": {"success": False, "images": []},
            "playwright_method": {"success": False, "images": []},
            "combined_result": {"success": False, "images": []}
        }
        
        # 1. Requests 방식으로 테스트
        try:
            response = self.session.get(url, timeout=self.timeout_config)
            response.raise_for_status()
            
            soup = self._get_soup(response.text)
            product_data = self._extract_product_data(soup, product_idx, url)
            
            test_results["requests_method"]["success"] = True
            test_results["requests_method"]["images"] = product_data.get("image_gallery", [])
            
            self.logger.info(f"Requests method found {len(test_results['requests_method']['images'])} images")
            
        except Exception as e:
            self.logger.error(f"Requests method failed: {e}")
            
        # 2. Playwright 방식으로 테스트
        images_playwright = self._extract_images_with_playwright_common(url)
        self.logger.info(f"Test extraction with Playwright: found {len(images_playwright)} images")
        
        # 3. 결합된 결과
        combined_images = self._merge_unique_images(
            test_results["requests_method"]["images"], 
            images_playwright
        )
        
        test_results["combined_result"]["images"] = combined_images
        test_results["combined_result"]["success"] = bool(combined_images)
        
        self.logger.info(f"Combined methods found {len(combined_images)} images")
        
        # 테스트 결과 저장
        with open(f"{self.output_dir}/test_product_{product_idx}_results.txt", "w", encoding="utf-8") as f:
            f.write(f"Product URL: {url}\n\n")
            f.write(f"1. Requests Method (Images found: {len(test_results['requests_method']['images'])})\n")
            for i, img in enumerate(test_results["requests_method"]["images"]):
                f.write(f"  {i+1}. {img}\n")
            
            f.write(f"\n2. Playwright Method (Images found: {len(images_playwright)})\n")
            for i, img in enumerate(images_playwright):
                f.write(f"  {i+1}. {img}\n")
                
            f.write(f"\n3. Combined Result (Total unique images: {len(combined_images)})\n")
            for i, img in enumerate(combined_images):
                f.write(f"  {i+1}. {img}\n")
        
        return test_results

    def _extract_images_with_stubborn_method(self, url: str, product_name: str) -> List[str]:
        """
        까다로운 경우를 위한 특수 이미지 추출 방법.
        일반적인 방법으로 이미지를 찾을 수 없을 때 사용합니다.
        
        Args:
            url: 제품 페이지 URL
            product_name: 제품 이름 (확인용)
            
        Returns:
            추출된 이미지 URL 목록
        """
        self.logger.info(f"Using stubborn method to extract images for: {product_name}")
        images = []
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                
                # 확장된 viewport로 컨텍스트 생성 (더 많은 컨텐츠 로드)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent=self.user_agent
                )
                
                page = context.new_page()
                
                # 대화 상자 처리
                page.on("dialog", self._handle_dialog)
                
                # 타임아웃 연장하여 더 오래 기다리기
                page.goto(url, timeout=60000, wait_until="networkidle")
                
                # 페이지가 완전히 로드될 때까지 기다리기
                self._wait_for_load(page)
                
                # 모든 이미지 요소 수집
                images = page.evaluate("""() => {
                    function isValidImage(url) {
                        return url && 
                               url.trim() !== '' && 
                               !url.includes('btn_') &&
                               !url.includes('icon_') &&
                               url.match(/\\.(jpg|jpeg|png|gif)(\\?|$)/i);
                    }
                    
                    // 모든 이미지 수집
                    const allImages = Array.from(document.querySelectorAll('img'))
                        .map(img => img.src)
                        .filter(isValidImage);
                    
                    // 큰 이미지만 수집 (getComputedStyle 사용)
                    const largeImages = Array.from(document.querySelectorAll('img')).filter(img => {
                        const style = window.getComputedStyle(img);
                        const width = parseInt(style.width);
                        const height = parseInt(style.height);
                        return (width > 150 || height > 150) && isValidImage(img.src);
                    }).map(img => img.src);
                    
                    // 백그라운드 이미지 URL 수집
                    const bgElements = Array.from(document.querySelectorAll('*')).filter(el => {
                        const style = window.getComputedStyle(el);
                        const bgImage = style.backgroundImage || '';
                        return bgImage.includes('url(') && !bgImage.includes('gradient');
                    });
                    
                    const bgImages = bgElements.map(el => {
                        const style = window.getComputedStyle(el);
                        const bgImage = style.backgroundImage;
                        const urlMatch = bgImage.match(/url\\(['"]?([^'"\\)]+)['"]?\\)/);
                        return urlMatch ? urlMatch[1] : null;
                    }).filter(url => isValidImage(url));
                    
                    // 결과 합치기 (중복 제거)
                    const uniqueImages = [...new Set([...largeImages, ...allImages, ...bgImages])];
                    return uniqueImages;
                }""")
                
                # 스크린샷 저장 (분석용)
                try:
                    screenshot_path = f"{self.output_dir}/stubborn_{hashlib.md5(url.encode()).hexdigest()[:8]}.png"
                    page.screenshot(path=screenshot_path)
                    self.logger.info(f"Saved screenshot to {screenshot_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save screenshot: {e}")
                    
                # 이미지 요소가 없는 경우, 스크롤 후 다시 시도
                if not images:
                    self.logger.info("No images found on initial load, scrolling and retrying...")
                    
                    # 페이지 전체 스크롤
                    page.evaluate("""() => {
                        return new Promise((resolve) => {
                            let totalHeight = 0;
                            const distance = 100;
                            const timer = setInterval(() => {
                                const scrollHeight = document.body.scrollHeight;
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                
                                if(totalHeight >= scrollHeight) {
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 100);
                        });
                    }""")
                    
                    # 잠시 대기 후 다시 이미지 검색
                    page.wait_for_timeout(2000)
                    
                    # 다시 이미지 수집
                    images = page.evaluate("""() => {
                        const allImages = Array.from(document.querySelectorAll('img'))
                            .map(img => img.src)
                            .filter(url => url && 
                                   url.trim() !== '' && 
                                   !url.includes('btn_') &&
                                   !url.includes('icon_'));
                        return [...new Set(allImages)];
                    }""")
                
                # 마지막 수단: 이미지 검색을 위한 DOM 검사
                if not images:
                    self.logger.info("Still no images, trying DOM inspection...")
                    html_content = page.content()
                    
                    # HTML 내용에서 이미지 URL 패턴 검색
                    img_patterns = [
                        r'src=[\'"]([^"\']+\.(jpg|jpeg|png|gif))[\'"]',
                        r'url\([\'"]?([^"\'()]+\.(jpg|jpeg|png|gif))[\'"]?\)',
                        r'background(-image)?:[^;]+url\([\'"]?([^"\'()]+)[\'"]?\)'
                    ]
                    
                    for pattern in img_patterns:
                        matches = re.findall(pattern, html_content)
                        if matches:
                            for match in matches:
                                img_url = match[0] if isinstance(match, tuple) else match
                                img_url = urljoin(url, img_url)
                                if img_url not in images:
                                    images.append(img_url)

                browser.close()
        
        except Exception as e:
            self.logger.error(f"Stubborn image extraction failed: {e}", exc_info=True)
        
        # 추출된 이미지 필터링 및 정리
        filtered_images = []
        for img_url in images:
            # 유효한 이미지 URL인지 확인 (상대 URL 처리)
            if img_url:
                try:
                    full_url = urljoin(url, img_url)
                    
                    # 작은 아이콘, 버튼 등 필터링
                    if not any(x in full_url.lower() for x in ['icon', 'button', 'btn_', 'pixel.gif']):
                        filtered_images.append(full_url)
                except Exception:
                    continue
        
        self.logger.info(f"Stubborn method found {len(filtered_images)} images")
        return filtered_images

    def _extract_images_with_playwright_common(self, url: str) -> list:
        self.logger.info("Extracting images using shared Playwright method")
        images = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url)
                page.wait_for_load_state("networkidle")
                images = page.evaluate("Array.from(document.images).map(img => img.src)")
                browser.close()
        except Exception as e:
            self.logger.error(f"Error in shared Playwright extraction: {e}")
        return images

    def _wait_for_load(self, page):
        """페이지가 완전히 로드될 때까지 기다림"""
        try:
            # 네트워크 유휴 상태 대기
            page.wait_for_load_state("networkidle", timeout=30000)
            
            # DOM 콘텐츠 로드 대기
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            
            # 잠시 기다림 (추가 리소스 로드를 위해)
            page.wait_for_timeout(2000)
            
        except Exception as e:
            self.logger.warning(f"Wait for load timed out: {e}")

    def _merge_unique_images(self, *image_lists: list) -> list:
        return list(dict.fromkeys(img for lst in image_lists for img in lst))

    def _extract_detailed_images_with_playwright(self, url: str) -> Tuple[List[str], str]:
        """
        Playwright를 사용하여 상세한 이미지 추출을 수행합니다.
        
        Args:
            url: 제품 페이지 URL
            
        Returns:
            추출된 이미지 URL 목록과 HTML 콘텐츠
        """
        self.logger.info("Extracting images using detailed Playwright method")
        images = []
        html_content = ""
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url, wait_until="networkidle")
                
                # 페이지가 완전히 로드될 때까지 기다리기
                self._wait_for_load(page)
                
                # 모든 이미지 요소 수집
                images = page.evaluate("""() => {
                    function isValidImage(url) {
                        return url && 
                               url.trim() !== '' && 
                               !url.includes('btn_') &&
                               !url.includes('icon_') &&
                               url.match(/\\.(jpg|jpeg|png|gif)(\\?|$)/i);
                    }
                    
                    // 모든 이미지 수집
                    const allImages = Array.from(document.querySelectorAll('img'))
                        .map(img => img.src)
                        .filter(isValidImage);
                    
                    // 큰 이미지만 수집 (getComputedStyle 사용)
                    const largeImages = Array.from(document.querySelectorAll('img')).filter(img => {
                        const style = window.getComputedStyle(img);
                        const width = parseInt(style.width);
                        const height = parseInt(style.height);
                        return (width > 150 || height > 150) && isValidImage(img.src);
                    }).map(img => img.src);
                    
                    // 백그라운드 이미지 URL 수집
                    const bgElements = Array.from(document.querySelectorAll('*')).filter(el => {
                        const style = window.getComputedStyle(el);
                        const bgImage = style.backgroundImage || '';
                        return bgImage.includes('url(') && !bgImage.includes('gradient');
                    });
                    
                    const bgImages = bgElements.map(el => {
                        const style = window.getComputedStyle(el);
                        const bgImage = style.backgroundImage;
                        const urlMatch = bgImage.match(/url\\(['"]?([^'"\\)]+)['"]?\\)/);
                        return urlMatch ? urlMatch[1] : null;
                    }).filter(url => isValidImage(url));
                    
                    // 결과 합치기 (중복 제거)
                    const uniqueImages = [...new Set([...largeImages, ...allImages, ...bgImages])];
                    return uniqueImages;
                }""")
                
                # 스크린샷 저장 (분석용)
                try:
                    screenshot_path = f"{self.output_dir}/detailed_{hashlib.md5(url.encode()).hexdigest()[:8]}.png"
                    page.screenshot(path=screenshot_path)
                    self.logger.info(f"Saved screenshot to {screenshot_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save screenshot: {e}")
                    
                # 이미지 요소가 없는 경우, 스크롤 후 다시 시도
                if not images:
                    self.logger.info("No images found on initial load, scrolling and retrying...")
                    
                    # 페이지 전체 스크롤
                    page.evaluate("""() => {
                        return new Promise((resolve) => {
                            let totalHeight = 0;
                            const distance = 100;
                            const timer = setInterval(() => {
                                const scrollHeight = document.body.scrollHeight;
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                
                                if(totalHeight >= scrollHeight) {
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 100);
                        });
                    }""")
                    
                    # 잠시 대기 후 다시 이미지 검색
                    page.wait_for_timeout(2000)
                    
                    # 다시 이미지 수집
                    images = page.evaluate("""() => {
                        const allImages = Array.from(document.querySelectorAll('img'))
                            .map(img => img.src)
                            .filter(url => url && 
                                   url.trim() !== '' && 
                                   !url.includes('btn_') &&
                                   !url.includes('icon_'));
                        return [...new Set(allImages)];
                    }""")
                
                # 마지막 수단: 이미지 검색을 위한 DOM 검사
                if not images:
                    self.logger.info("Still no images, trying DOM inspection...")
                    html_content = page.content()
                    
                    # HTML 내용에서 이미지 URL 패턴 검색
                    img_patterns = [
                        r'src=[\'"]([^"\']+\.(jpg|jpeg|png|gif))[\'"]',
                        r'url\([\'"]?([^"\'()]+\.(jpg|jpeg|png|gif))[\'"]?\)',
                        r'background(-image)?:[^;]+url\([\'"]?([^"\'()]+)[\'"]?\)'
                    ]
                    
                    for pattern in img_patterns:
                        matches = re.findall(pattern, html_content)
                        if matches:
                            for match in matches:
                                img_url = match[0] if isinstance(match, tuple) else match
                                img_url = urljoin(url, img_url)
                                if img_url not in images:
                                    images.append(img_url)

                browser.close()
        
        except Exception as e:
            self.logger.error(f"Detailed image extraction failed: {e}", exc_info=True)
        
        # 추출된 이미지 필터링 및 정리
        filtered_images = []
        for img_url in images:
            # 유효한 이미지 URL인지 확인 (상대 URL 처리)
            if img_url:
                try:
                    full_url = urljoin(url, img_url)
                    
                    # 작은 아이콘, 버튼 등 필터링
                    if not any(x in full_url.lower() for x in ['icon', 'button', 'btn_', 'pixel.gif']):
                        filtered_images.append(full_url)
                except Exception:
                    continue
        
        self.logger.info(f"Detailed method found {len(filtered_images)} images")
        return filtered_images, html_content
