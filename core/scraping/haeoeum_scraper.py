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
import sys
import asyncio

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

from core.data_models import Product
from core.scraping.base_multi_layer_scraper import BaseMultiLayerScraper
from core.scraping.utils import extract_main_image


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
        max_retries: Optional[int] = None,
        cache: Optional[Any] = None,
        timeout: Optional[int] = None,
        connect_timeout: Optional[int] = None,
        read_timeout: Optional[int] = None,
        backoff_factor: Optional[float] = None,
        use_proxies: bool = False,
        debug: bool = False,
        output_dir: str = "output",
        headless: Optional[bool] = None,
        user_agent: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ):
        """스크래퍼 초기화"""
        # Load configuration
        self.config = self._load_config()
        
        # Use provided values or fall back to config values
        self.max_retries = max_retries or int(self.config.get('SCRAPING', 'max_retries', fallback='3'))
        self.timeout = timeout or int(self.config.get('SCRAPING', 'extraction_timeout', fallback='15'))
        self.connect_timeout = connect_timeout or int(self.config.get('SCRAPING', 'connect_timeout', fallback='5'))
        self.read_timeout = read_timeout or int(self.config.get('SCRAPING', 'read_timeout', fallback='10'))
        self.backoff_factor = backoff_factor or float(self.config.get('SCRAPING', 'backoff_factor', fallback='0.3'))
        self.headless = headless if headless is not None else self.config.getboolean('SCRAPING', 'headless', fallback=True)
        self.user_agent = user_agent or self.config.get('SCRAPING', 'user_agent', fallback='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        self.cache_ttl = cache_ttl or int(self.config.get('SCRAPING', 'cache_ttl', fallback='3600'))
        
        # BaseScraper 초기화
        super().__init__(max_retries=self.max_retries, timeout=self.timeout, cache=cache)

        # --- Base Attributes ---
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.output_dir = output_dir
        self.timeout_config = (self.connect_timeout, self.read_timeout)
        self.playwright_available = PLAYWRIGHT_AVAILABLE

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # Playwright 설정 최적화
        if PLAYWRIGHT_AVAILABLE:
            self.playwright_config = {
                "viewport": {
                    "width": int(self.config.get('SCRAPING', 'viewport_width', fallback='1280')),
                    "height": int(self.config.get('SCRAPING', 'viewport_height', fallback='720'))
                },
                "timeout": int(self.config.get('SCRAPING', 'wait_timeout', fallback='60000')),  # 타임아웃을 60초로 증가
                "wait_until": "domcontentloaded",  # 더 빠른 페이지 로드 조건
                "ignore_https_errors": not self.config.getboolean('SCRAPING', 'ssl_verification', fallback=True)
            }
        else:
            self.playwright_available = False
            self.logger.warning("Playwright is not installed. Some features will be limited.")

        # requests 세션 최적화
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[int(x) for x in self.config.get('SCRAPING', 'retry_on_specific_status', fallback='429,500,502,503,504').split(',')],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=int(self.config.get('SCRAPING', 'connection_pool_size', fallback='20')),
            pool_maxsize=int(self.config.get('SCRAPING', 'connection_pool_size', fallback='20')),
            pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": self.user_agent})

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
                "selector": "img#target_img, img[style*='cursor:hand'][onclick*='view_big'], img[width='330'][height='330']",
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
            "image_url": re.compile(r'url\([\'"]?([^"\'()]+\.(jpg|jpeg|png|gif))[\'"]?\)', re.IGNORECASE),
            "onclick_image": re.compile(r"view_big\('([^']+)'", re.IGNORECASE),
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
        """
        해오름 기프트 상품 페이지에서 상품 정보를 추출합니다.

        Args:
            product_idx: 상품 ID (p_idx 파라미터 값)

        Returns:
            Product 객체 또는 None (실패 시)
        """
        # 캐시 확인
        cache_key = f"haeoeum_product|{product_idx}"
        cached_result = self.get_cached_data(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 상품 데이터 사용: p_idx={product_idx}")
            return Product.from_dict(cached_result)

        url = f"{self.PRODUCT_VIEW_URL}?p_idx={product_idx}"
        self.logger.info(f"상품 이미지 추출 시작: {url}")

        # 기본 상품 정보 설정 (Excel에서 실제 정보를 가져오므로 최소한만 설정)
        product_data = {
            "id": hashlib.md5(f"haeoeum_{product_idx}".encode()).hexdigest(),
            "name": f"JCL_{product_idx}",  # Excel에서 실제 이름을 사용
            "source": "haeoeum",
            "price": 0,  # Excel에서 실제 가격을 사용
            "url": url,
            "image_url": "",
            "image_gallery": [],
            "product_code": product_idx,
            "status": "OK"
        }
        
        # 1. requests로 이미지 추출 시도
        try:
            response = self.session.get(url, timeout=self.timeout_config)
            response.raise_for_status()
            
            # 인코딩 처리
            if not response.encoding or response.encoding == 'ISO-8859-1':
                try:
                    import chardet
                    detected = chardet.detect(response.content)
                    if detected['confidence'] > 0.7:
                        response.encoding = detected['encoding']
                    else:
                        response.encoding = 'cp949'
                except ImportError:
                    response.encoding = 'cp949'
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 이미지 추출
            images = self._extract_images_with_soup(soup, url)
            if images:
                product_data["image_url"] = images[0]
                product_data["image_gallery"] = images
                self.logger.info(f"requests로 이미지 {len(images)}개 추출 성공")
                
        except Exception as e:
            self.logger.error(f"requests를 사용한 이미지 추출 실패: {e}")

        # 2. Playwright로 시도 (requests 실패 시 또는 이미지가 없는 경우)
        if (not product_data.get("image_gallery") or len(product_data["image_gallery"]) == 0) and self.playwright_available:
            try:
                images = self._extract_images_with_playwright_common(url)
                if images:
                    product_data["image_url"] = images[0]
                    product_data["image_gallery"] = images
                    self.logger.info(f"Playwright로 이미지 {len(images)}개 추출 성공")
            except Exception as e:
                self.logger.error(f"Playwright를 사용한 이미지 추출 실패: {e}")

        # 3. 대체 방법으로 시도 (이전 방법 모두 실패 시)
        if not product_data.get("image_gallery") or len(product_data["image_gallery"]) == 0:
            try:
                images = self._extract_images_with_stubborn_method(url)
                if images:
                    product_data["image_url"] = images[0]
                    product_data["image_gallery"] = images
                    self.logger.info(f"대체 방법으로 이미지 {len(images)}개 추출 성공")
            except Exception as e:
                self.logger.error(f"대체 방법으로 이미지 추출 실패: {e}")

        # 4. 이미지가 없는 경우 기본 이미지 설정
        if not product_data.get("image_gallery") or len(product_data["image_gallery"]) == 0:
            self.logger.warning(f"이미지 추출 실패, 기본 이미지 사용: {product_idx}")
            product_data["image_url"] = f"{self.BASE_URL}/images/no_image.jpg"
            product_data["image_gallery"] = [f"{self.BASE_URL}/images/no_image.jpg"]
            product_data["status"] = "Image Not Found"
        
        # 5. Product 객체 생성
        try:
            product = Product(**{k: v for k, v in product_data.items() if k in Product.__annotations__})
            
            # 이미지 갤러리 설정
            product.image_gallery = product_data['image_gallery']
            
            # 원본 데이터 저장
            product.original_input_data = product_data
            
            # 캐시에 저장
            self.cache_sparse_data(cache_key, product.to_dict())
            
            return product
            
        except Exception as e:
            self.logger.error(f"Product 객체 생성 실패: {e}")
            return None

    def _create_fallback_product(self, product_idx: str, url: str) -> Product:
        """
        문제 발생 시 대체 상품 객체 생성
        해오름 상품은 항상 존재한다는 대전제에 따라 필요한 기본 상품 객체 생성
        
        Parameters:
            product_idx: 상품 ID
            url: 상품 URL
            
        Returns:
            기본 설정의 Product 객체
        """
        # 기본 상품 데이터 생성
        fallback_data = {
            "id": hashlib.md5(f"haeoeum_{product_idx}".encode()).hexdigest(),
            "name": f"해오름상품_{product_idx}",
            "source": "haeoeum",
            "price": 10000,  # 임시 가격
            "url": url,
            "image_url": f"{self.BASE_URL}/images/no_image.jpg",
            "status": "Fallback",
            "product_code": product_idx,
            "image_gallery": [f"{self.BASE_URL}/images/no_image.jpg"],
            "quantity_prices": {"1": 10000}
        }
        
        # 상품 객체 생성
        product = Product(**{k: v for k, v in fallback_data.items() if k in Product.__annotations__})
        product.image_gallery = fallback_data["image_gallery"]
        product.quantity_prices = fallback_data["quantity_prices"]
        product.original_input_data = fallback_data
        
        self.logger.warning(f"대체 상품 객체 생성: {product_idx}")
        return product

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
        """
        상품 상세 페이지에서 필요한 정보를 추출합니다.
        
        Args:
            soup: BeautifulSoup 객체
            product_idx: 상품 ID
            url: 상품 URL
            
        Returns:
            추출된 상품 정보 딕셔너리
        """
        product_data = {
            "id": hashlib.md5(f"haeoeum_{product_idx}".encode()).hexdigest(),
            "source": "haeoeum",
            "url": url,
            "name": "",
            "price": 0,
            "image_url": "",
            "image_gallery": [],
            "product_code": product_idx,
            "status": "Processing",
            "specifications": {},
            "quantity_prices": {},
            "description": "",
        }

        try:
            # 1. 상품명 추출
            title_element = soup.select_one(self.selectors["product_title"]["selector"])
            if title_element:
                title = title_element.text.strip()
                product_data["name"] = self._normalize_text(title)
                self.logger.debug(f"상품명 추출: {product_data['name']}")
            
            if not product_data["name"]:
                product_data["name"] = f"해오름상품_{product_idx}"
                self.logger.warning(f"상품명 추출 실패, 임시 이름 사용: {product_data['name']}")

            # 2. 상품 코드 추출
            code_element = soup.select_one(self.selectors["product_code"]["selector"])
            if code_element:
                product_data["product_code"] = code_element.text.strip()
            else:
                # 텍스트에서 상품 코드 패턴 찾기
                text_content = soup.get_text()
                code_match = self.patterns["product_code"].search(text_content)
                if code_match:
                    product_data["product_code"] = code_match.group(1)

            # 3. 가격 추출
            price = 0
            # 3.1 적용 가격 확인
            price_element = soup.select_one(self.selectors["applied_price"]["selector"])
            if price_element:
                price_text = price_element.text.strip()
                price_match = self.patterns["price_number"].search(price_text)
                if price_match:
                    price = int(price_match.group().replace(",", ""))

            # 3.2 총 가격 확인
            if price == 0:
                total_price_element = soup.select_one(self.selectors["total_price"]["selector"])
                if total_price_element:
                    price_text = total_price_element.text.strip()
                    price_match = self.patterns["price_number"].search(price_text)
                    if price_match:
                        price = int(price_match.group().replace(",", ""))

            # 3.3 VAT 포함 여부 확인 및 처리
            text_content = soup.get_text().lower()
            vat_included = bool(self.patterns["vat_included"].search(text_content))
            if not vat_included and price > 0:
                price = int(price * 1.1)  # VAT 10% 추가

            product_data["price"] = price or 10000  # 가격이 0이면 임시 가격 설정

            # 4. 수량별 가격 추출
            quantity_prices = {}
            
            # 4.1 가격표에서 추출
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
                            if not vat_included:
                                price = int(price * 1.1)
                            quantity_prices[str(qty)] = price

            # 4.2 드롭다운에서 추출
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
                            if not vat_included:
                                price = int(price * 1.1)
                            quantity_prices[str(qty)] = price

            # 4.3 텍스트에서 수량별 가격 정보 찾기
            if not quantity_prices:
                matches = self.patterns["quantity_price"].findall(text_content)
                for match in matches:
                    qty = int(match[0])
                    price = int(match[1].replace(",", ""))
                    if not vat_included:
                        price = int(price * 1.1)
                    quantity_prices[str(qty)] = price

            # 4.4 기본 수량 가격 추가
            if not quantity_prices and product_data["price"] > 0:
                quantity_prices["1"] = product_data["price"]

            product_data["quantity_prices"] = quantity_prices

            # 5. 사양 정보 추출
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
                            product_data["specifications"][key] = value

            # 6. 제품 설명 추출
            desc_elements = soup.select("div.product_view_img, div.prd_detail, div.item_detail")
            if desc_elements:
                description = "\n".join([elem.text.strip() for elem in desc_elements])
                product_data["description"] = description

            # 7. 품절 여부 확인
            sold_out_element = soup.select_one(self.selectors["sold_out"]["selector"])
            is_sold_out = bool(sold_out_element)
            
            if not is_sold_out:
                if "품절" in text_content or "sold out" in text_content:
                    is_sold_out = True

            product_data["is_sold_out"] = is_sold_out
            product_data["status"] = "Sold Out" if is_sold_out else "OK"

        except Exception as e:
            self.logger.error(f"상품 데이터 추출 중 오류: {e}")
            product_data["status"] = "Error"

        return product_data

    def _normalize_text(self, text: str) -> str:
        """
        텍스트 정규화 및 인코딩 문제를 해결합니다.
        
        Args:
            text: 정규화할 텍스트
            
        Returns:
            정규화된 텍스트
        """
        if not text:
            return ""
            
        # 인코딩 문제 해결 - CP949/EUC-KR 인코딩 문제 대응
        try:
            # 이미 깨진 문자열이 들어왔을 경우 복원 시도
            if any(c in text for c in ['¿', 'À', 'Á', '¾', '½']):
                # CP949로 인코딩된 바이트 배열로 가정하고 복원 시도
                encoded_bytes = text.encode('latin1')
                text = encoded_bytes.decode('cp949', errors='replace')
        except Exception as e:
            self.logger.warning(f"텍스트 정규화 중 오류: {e}")
            
        # 공백 및 특수문자 정리
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_product_idx(self, url: str) -> Optional[str]:
        """
        URL에서 제품 ID(p_idx)를 추출합니다.
        
        Args:
            url: 제품 URL
            
        Returns:
            제품 ID 또는 URL 분석 실패 시 None
        """
        try:
            if not url or "product_view.asp" not in url:
                return None
                
            # URL 파싱
            parsed_url = urlparse(url)
            query_params = dict(param.split('=') for param in parsed_url.query.split('&'))
            
            # p_idx 파라미터 찾기
            return query_params.get('p_idx')
        except Exception as e:
            self.logger.error(f"URL에서 제품 ID 추출 실패: {url} - {e}")
            return None
            
    # 이전 메서드의 별칭 유지 (하위 호환성)
    extract_product_idx_from_url = _extract_product_idx

    def _handle_dialog(self, dialog):
        """
        대화 상자를 처리합니다.
        품절 등 상태 메시지를 확인하는 데 사용됩니다.
        
        Args:
            dialog: Playwright 대화 상자 객체
        """
        self.dialog_message = dialog.message
        self.logger.debug(f"대화 상자 메시지: {dialog.message}")
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
        """
        상품 URL에서 재고 상태를 확인합니다.
        
        Args:
            url: 상품 URL
            
        Returns:
            (재고 있음 여부, 상태 메시지) 튜플
        """
        try:
            # requests로 먼저 시도
            response = self.session.get(url, timeout=self.timeout_config)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 품절 표시 확인
            sold_out_element = soup.select_one(self.selectors["sold_out"]["selector"])
            if sold_out_element:
                return False, sold_out_element.text.strip() or "품절"
            
            # 텍스트에서 품절 키워드 확인
            text_content = soup.get_text().lower()
            if "품절" in text_content or "sold out" in text_content:
                return False, "품절"
            
            # Playwright로 추가 확인 (필요한 경우)
            if self.playwright_available:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self.headless)
                    page = browser.new_page()
                    page.on("dialog", self._handle_dialog)
                    
                    page.goto(url, wait_until=self.playwright_config["wait_until"])
                    
                    # 대화 상자 메시지 확인
                    if self.dialog_message and any(keyword in self.dialog_message.lower() for keyword in ["품절", "재고", "sold out"]):
                        return False, self.dialog_message
                    
                    # 품절 요소 확인
                    sold_out = page.locator(self.selectors["sold_out"]["selector"]).count() > 0
                    if sold_out:
                        return False, "품절"
                    
                    browser.close()
            
            return True, "재고 있음"
            
        except Exception as e:
            self.logger.error(f"재고 상태 확인 중 오류: {e}")
            return False, f"확인 실패: {str(e)}"

    def search_product(self, query: str, max_items: int = 50) -> List[Product]:
        """
        해오름 기프트에서 상품을 검색합니다.

        Args:
            query: 검색어
            max_items: 최대 검색 결과 수

        Returns:
            List[Product]: 검색된 상품 목록
        """
        self.logger.info(f"해오름 기프트에서 '{query}' 검색 시작")
        
        # 캐시 확인
        cache_key = f"haeoeum_search|{query}"
        cached_result = self.get_cached_data(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 검색 결과 사용: '{query}'")
            return [Product.from_dict(p) for p in cached_result]

        products = []
        try:
            # 검색 URL 구성
            search_url = f"{self.BASE_URL}/product/product_list.asp"
            params = {
                "search_keyword": query,
                "search_type": "all"
            }
            
            # requests로 먼저 시도
            try:
                response = self.session.get(search_url, params=params, verify=False)
                response.raise_for_status()
                
                # 인코딩 처리
                if not response.encoding or response.encoding == 'ISO-8859-1':
                    try:
                        import chardet
                        detected = chardet.detect(response.content)
                        if detected['confidence'] > 0.7:
                            response.encoding = detected['encoding']
                        else:
                            response.encoding = 'cp949'
                    except ImportError:
                        response.encoding = 'cp949'
                
                soup = BeautifulSoup(response.text, "html.parser")
                product_elements = soup.select(self.selectors["product_list"]["selector"])
                
                if product_elements:
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
                
            except Exception as e:
                self.logger.error(f"requests를 사용한 검색 실패: {e}")

            # Playwright로 시도 (requests 실패 시)
            if not products and self.playwright_available:
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=self.headless)
                        page = browser.new_page()
                        
                        # 검색 페이지 로드
                        page.goto(search_url, params=params, wait_until=self.playwright_config["wait_until"])
                        
                        # 상품 목록 대기
                        page.wait_for_selector(self.selectors["product_list"]["selector"], timeout=5000)
                        
                        # HTML 파싱
                        html = page.content()
                        soup = BeautifulSoup(html, "html.parser")
                        product_elements = soup.select(self.selectors["product_list"]["selector"])
                        
                        if product_elements:
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
                        
                        browser.close()
                        
                except Exception as e:
                    self.logger.error(f"Playwright를 사용한 검색 실패: {e}")

            # 검색 결과가 없으면 "동일상품 없음" 반환
            if not products:
                self.logger.warning(f"❌ 상품이 존재하지 않음: '{query}' 검색 결과 없음")
                no_match_product = Product(
                    id=hashlib.md5(f"no_match_haeoeum_{query}".encode()).hexdigest(),
                    name=f"동일상품 없음 - {query}",
                    source="haeoeum",
                    price=0,
                    url="",
                    image_url=f"{self.BASE_URL}/images/no_image.jpg",
                    status="Not Found"
                )
                products = [no_match_product]
            
            # 캐시에 저장
            self.cache_sparse_data(cache_key, [p.to_dict() for p in products])
            
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

    def _extract_images(self, url: str, soup: Optional[BeautifulSoup] = None) -> List[str]:
        """
        상품 페이지에서 이미지를 추출합니다.
        여러 방법을 순차적으로 시도하여 최대한 많은 이미지를 찾습니다.

        Args:
            url: 상품 페이지 URL
            soup: 이미 파싱된 BeautifulSoup 객체 (선택적)

        Returns:
            List[str]: 추출된 이미지 URL 목록
        """
        images = []
        
        # 1. BeautifulSoup으로 먼저 시도
        if soup:
            soup_images = self._extract_images_with_soup(soup, url)
            self.logger.info(f"BeautifulSoup으로 {len(soup_images)}개 이미지 추출")
            images.extend(soup_images)
            
        # 2. Playwright로 시도 (이미지가 없거나 Playwright가 사용 가능한 경우)
        if (not images or len(images) < 2) and self.playwright_available:
            try:
                playwright_images = self._extract_images_with_playwright(url)
                self.logger.info(f"Playwright로 {len(playwright_images)}개 이미지 추출")
                images.extend(playwright_images)
            except Exception as e:
                self.logger.warning(f"Playwright 이미지 추출 실패: {e}")
        
        # 3. 이미지가 여전히 없으면 마지막 수단으로 시도
        if not images and self.playwright_available:
            try:
                stubborn_images = self._extract_images_with_stubborn_method(url)
                self.logger.info(f"대체 방법으로 {len(stubborn_images)}개 이미지 추출 성공")
                images.extend(stubborn_images)
            except Exception as e:
                self.logger.warning(f"대체 이미지 추출 실패: {e}")
        
        # 중복 제거 및 URL 정규화
        unique_images = []
        for img_url in images:
            if img_url:
                try:
                    full_url = urljoin(url, img_url)
                    if full_url.strip() and not any(x in full_url.lower() for x in ['icon', 'button', 'btn_', 'pixel.gif']):
                        if full_url not in unique_images:
                            unique_images.append(full_url)
                except Exception:
                    continue
        
        # 이미지가 없으면 기본 이미지 반환
        if not unique_images:
            default_image = f"{self.BASE_URL}/images/no_image.jpg"
            unique_images.append(default_image)
            
        return unique_images

    def _extract_images_with_soup(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """BeautifulSoup을 사용한 이미지 추출"""
        images = []
        
        # 1. 메인 이미지 추출
        main_image = soup.select_one(self.selectors["main_image"]["selector"])
        if main_image and main_image.get('src'):
            img_url = main_image.get('src')
            if img_url:
                images.append(img_url)
                
            # onclick 속성에서 큰 이미지 URL 추출
            onclick = main_image.get('onclick', '')
            if onclick:
                onclick_match = self.patterns["onclick_image"].search(onclick)
                if onclick_match:
                    big_img_url = onclick_match.group(1)
                    if big_img_url:
                        images.append(big_img_url)
        
        # 2. 대체 메인 이미지 추출
        alt_image = soup.select_one(self.selectors["alt_main_image"]["selector"])
        if alt_image and alt_image.get('src'):
            images.append(alt_image.get('src'))
        
        # 3. 썸네일 이미지 추출
        thumbnails = soup.select(self.selectors["thumbnail_images"]["selector"])
        for thumb in thumbnails:
            if thumb.get('src'):
                images.append(thumb.get('src'))
        
        # 4. 상품 설명 이미지 추출
        desc_images = soup.select(self.selectors["description_images"]["selector"])
        for img in desc_images:
            if img.get('src'):
                images.append(img.get('src'))
        
        # 5. style 속성에서 배경 이미지 URL 추출
        for element in soup.select('[style*="background"]'):
            style = element.get('style', '')
            matches = self.patterns["image_url"].findall(style)
            for match in matches:
                if match[0]:  # match[0]는 전체 URL
                    images.append(match[0])
        
        return images

    def _extract_images_with_playwright(self, url: str) -> List[str]:
        """Playwright를 사용하여 이미지 추출 (동기 방식으로 변경)"""
        if not self.playwright_available:
            return []

        max_retries = 3
        retry_delay = 2  # 초
        images = []

        for attempt in range(max_retries):
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self.headless)
                    page = browser.new_page()
                    
                    # 타임아웃 설정
                    page.set_default_timeout(self.playwright_config["timeout"])
                    
                    # 페이지 로드
                    page.goto(url, wait_until=self.playwright_config["wait_until"])
                    
                    # 이미지 로드 대기
                    page.wait_for_selector("img", timeout=5000)
                    
                    # 이미지 추출
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
                        
                        // 큰 이미지만 수집
                        const largeImages = Array.from(document.querySelectorAll('img')).filter(img => {
                            const width = img.naturalWidth || img.width;
                            const height = img.naturalHeight || img.height;
                            return (width > 150 || height > 150) && isValidImage(img.src);
                        }).map(img => img.src);
                        
                        return [...new Set([...largeImages, ...allImages])];
                    }""")
                    
                    browser.close()
                    
                    if images:
                        self.logger.info(f"Playwright로 {len(images)}개 이미지 추출 성공 (시도 {attempt+1}/{max_retries})")
                        return images
                    
            except Exception as e:
                self.logger.warning(f"Playwright 이미지 추출 시도 {attempt + 1}/{max_retries} 실패: {e}")
                time.sleep(retry_delay)
        
        self.logger.error(f"Playwright 이미지 추출 최종 실패")
        return []

    def _extract_images_with_stubborn_method(self, url: str) -> List[str]:
        """까다로운 경우를 위한 특수 이미지 추출 방법"""
        images = []
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent=self.user_agent
                )
                
                page = context.new_page()
                page.on("dialog", self._handle_dialog)
                
                # 페이지 로드 및 대기
                page.goto(url, timeout=60000, wait_until="networkidle")
                page.wait_for_selector("img", timeout=10000)
                
                # 전체 페이지 스크롤
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
                
                # 잠시 대기
                page.wait_for_timeout(2000)
                
                # 이미지 추출
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
                    
                    // 큰 이미지만 수집
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
                    
                    // onclick 속성에서 이미지 URL 추출
                    const onclickImages = Array.from(document.querySelectorAll('[onclick*="view_big"]'))
                        .map(el => {
                            const onclick = el.getAttribute('onclick');
                            const match = onclick.match(/view_big\\(['"]([^'"]+)['"]\\)/);
                            return match ? match[1] : null;
                        })
                        .filter(url => isValidImage(url));
                    
                    return [...new Set([...largeImages, ...allImages, ...bgImages, ...onclickImages])];
                }""")
                
                browser.close()
                
        except Exception as e:
            self.logger.error(f"Stubborn 이미지 추출 실패: {e}")
            
        return images

    def _extract_images_with_playwright_common(self, url: str) -> list:
        self.logger.info("Extracting images using shared Playwright method")
        images = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                page = browser.new_page()
                page.goto(url)
                page.wait_for_load_state("networkidle")
                page.wait_for_selector("img", timeout=5000)
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
                    
                    // 큰 이미지만 수집
                    const largeImages = Array.from(document.querySelectorAll('img')).filter(img => {
                        const width = img.naturalWidth || img.width;
                        const height = img.naturalHeight || img.height;
                        return (width > 150 || height > 150) && isValidImage(img.src);
                    }).map(img => img.src);
                    
                    return [...new Set([...largeImages, ...allImages])];
                }""")
                browser.close()
        except Exception as e:
            self.logger.error(f"Error in shared Playwright extraction: {e}")
        return images

    def extract_images_for_excel_product(self, product_idx: str) -> Dict[str, Any]:
        """
        Excel에 포함된 제품의 이미지만 추출하는 전용 메서드.
        다른 정보는 Excel에서 이미 가져올 수 있기 때문에 이미지만 가져옵니다.
        
        Args:
            product_idx: 제품 ID (URL의 p_idx 파라미터)
            
        Returns:
            딕셔너리 { "image_url": 첫 번째 이미지 URL, "image_gallery": 이미지 URL 목록 }
        """
        url = f"{self.PRODUCT_VIEW_URL}?p_idx={product_idx}"
        self.logger.info(f"Excel 제품 이미지 추출 시작: {url}")
        
        # 캐시 확인
        cache_key = f"haeoeum_excel_image|{product_idx}"
        cached_result = self.get_cached_data(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 이미지 데이터 사용: p_idx={product_idx}")
            return cached_result
        
        result = {
            "image_url": f"{self.BASE_URL}/images/no_image.jpg",
            "image_gallery": [f"{self.BASE_URL}/images/no_image.jpg"],
            "status": "OK"
        }
        
        # 1. requests로 이미지 추출 시도
        try:
            response = self.session.get(url, timeout=self.timeout_config)
            response.raise_for_status()
            
            # 인코딩 처리
            if not response.encoding or response.encoding == 'ISO-8859-1':
                try:
                    import chardet
                    detected = chardet.detect(response.content)
                    if detected['confidence'] > 0.7:
                        response.encoding = detected['encoding']
                    else:
                        response.encoding = 'cp949'
                except ImportError:
                    response.encoding = 'cp949'
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 이미지 추출
            images = self._extract_images_with_soup(soup, url)
            if images:
                result["image_url"] = images[0]
                result["image_gallery"] = images
                self.logger.info(f"requests로 Excel 제품 이미지 {len(images)}개 추출 성공")
                
                # 캐시에 저장
                self.cache_sparse_data(cache_key, result)
                return result
                
        except Exception as e:
            self.logger.error(f"requests를 사용한 Excel 제품 이미지 추출 실패: {e}")

        # 2. Playwright로 시도
        if self.playwright_available:
            try:
                images = self._extract_images_with_playwright_common(url)
                if images:
                    result["image_url"] = images[0]
                    result["image_gallery"] = images
                    self.logger.info(f"Playwright로 Excel 제품 이미지 {len(images)}개 추출 성공")
                    
                    # 캐시에 저장
                    self.cache_sparse_data(cache_key, result)
                    return result
            except Exception as e:
                self.logger.error(f"Playwright를 사용한 Excel 제품 이미지 추출 실패: {e}")

        # 3. 대체 방법으로 시도
        try:
            images = self._extract_images_with_stubborn_method(url)
            if images:
                result["image_url"] = images[0]
                result["image_gallery"] = images
                self.logger.info(f"대체 방법으로 Excel 제품 이미지 {len(images)}개 추출 성공")
                
                # 캐시에 저장
                self.cache_sparse_data(cache_key, result)
                return result
        except Exception as e:
            self.logger.error(f"대체 방법으로 Excel 제품 이미지 추출 실패: {e}")

        # 이미지 추출 실패 시 결과 반환
        result["status"] = "Image Not Found"
        self.logger.warning(f"Excel 제품 이미지 추출 실패, 기본 이미지 사용: {product_idx}")
        
        # 캐시에 저장
        self.cache_sparse_data(cache_key, result)
        return result

    def get_product_image_url(self, product_idx: str) -> str:
        """
        주어진 제품의 이미지 URL만 반환하는 간단한 메서드.
        
        Args:
            product_idx: 제품 ID (URL의 p_idx 파라미터)
            
        Returns:
            이미지 URL 문자열
        """
        result = self.extract_images_for_excel_product(product_idx)
        return result["image_url"]
    
    def get_product_image_gallery(self, product_idx: str) -> List[str]:
        """
        주어진 제품의 이미지 갤러리 목록을 반환하는 간단한 메서드.
        
        Args:
            product_idx: 제품 ID (URL의 p_idx 파라미터)
            
        Returns:
            이미지 URL 목록
        """
        result = self.extract_images_for_excel_product(product_idx)
        return result["image_gallery"]
    
    def get_image_url_from_product_url(self, product_url: str) -> str:
        """
        제품 URL에서 직접 이미지 URL을 추출합니다.
        
        Args:
            product_url: 제품 전체 URL (예: http://www.jclgift.com/product/product_view.asp?p_idx=431692)
            
        Returns:
            이미지 URL (추출 실패 시 기본 이미지 URL 반환)
        """
        product_idx = self._extract_product_idx(product_url)
        if not product_idx:
            self.logger.warning(f"URL에서 제품 ID를 추출할 수 없음: {product_url}")
            return f"{self.BASE_URL}/images/no_image.jpg"
            
        return self.get_product_image_url(product_idx)
        
    def get_image_gallery_from_product_url(self, product_url: str) -> List[str]:
        """
        제품 URL에서 직접 이미지 갤러리를 추출합니다.
        
        Args:
            product_url: 제품 전체 URL (예: http://www.jclgift.com/product/product_view.asp?p_idx=431692)
            
        Returns:
            이미지 URL 목록 (추출 실패 시 기본 이미지 URL이 포함된 목록 반환)
        """
        product_idx = self._extract_product_idx(product_url)
        if not product_idx:
            self.logger.warning(f"URL에서 제품 ID를 추출할 수 없음: {product_url}")
            return [f"{self.BASE_URL}/images/no_image.jpg"]
            
        return self.get_product_image_gallery(product_idx)

    def batch_extract_images_from_urls(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        여러 제품 URL에서 이미지를 일괄 추출합니다.
        엑셀 데이터 처리에 최적화되어 있습니다.
        
        Args:
            urls: 제품 URL 목록
            
        Returns:
            딕셔너리 { "p_idx": { "image_url": 첫 번째 이미지 URL, "image_gallery": 이미지 URL 목록 } }
        """
        result = {}
        total_urls = len(urls)
        
        self.logger.info(f"URL {total_urls}개에서 이미지 일괄 추출 시작")
        
        for i, url in enumerate(urls):
            if not url or not isinstance(url, str):
                continue
                
            product_idx = self._extract_product_idx(url)
            if not product_idx:
                continue
                
            self.logger.info(f"[{i+1}/{total_urls}] 제품 이미지 추출 중: {product_idx}")
            images_data = self.extract_images_for_excel_product(product_idx)
            result[product_idx] = images_data
            
        self.logger.info(f"URL {total_urls}개 중 {len(result)}개 제품 이미지 추출 완료")
        return result
        
    def extract_images_from_excel_data(self, excel_data: List[Dict[str, Any]], url_column: str = "본사상품링크") -> List[Dict[str, Any]]:
        """
        엑셀 데이터에서 이미지를 추출하여 원본 데이터에 추가합니다.
        
        Args:
            excel_data: 엑셀에서 로드한 딕셔너리 목록 (각 행이 하나의 딕셔너리)
            url_column: URL이 포함된 열 이름
            
        Returns:
            이미지 URL이 추가된 원본 데이터 목록
        """
        if not excel_data:
            return []
            
        # URL 추출
        urls = [row.get(url_column, "") for row in excel_data if url_column in row]
        
        # 이미지 일괄 추출
        images_data = self.batch_extract_images_from_urls(urls)
        
        # 결과를 원본 데이터에 추가
        for row in excel_data:
            url = row.get(url_column, "")
            product_idx = self._extract_product_idx(url)
            
            if product_idx and product_idx in images_data:
                row["image_url"] = images_data[product_idx]["image_url"]
                row["image_gallery"] = images_data[product_idx]["image_gallery"]
                row["image_status"] = images_data[product_idx].get("status", "OK")
            else:
                # 이미지를 찾지 못한 경우 기본값 설정
                row["image_url"] = f"{self.BASE_URL}/images/no_image.jpg"
                row["image_gallery"] = [f"{self.BASE_URL}/images/no_image.jpg"]
                row["image_status"] = "URL Invalid or Not Found"
                
        return excel_data

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
                    
                    // onclick 속성에서 이미지 URL 추출
                    const onclickImages = Array.from(document.querySelectorAll('[onclick*="view_big"]'))
                        .map(el => {
                            const onclick = el.getAttribute('onclick');
                            const match = onclick.match(/view_big\\(['"]([^'"]+)['"]\\)/);
                            return match ? match[1] : null;
                        })
                        .filter(url => isValidImage(url));
                    
                    return [...new Set([...largeImages, ...allImages, ...bgImages, ...onclickImages])];
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

    def _merge_unique_images(self, *image_lists: list) -> list:
        """이미지 URL 목록들을 병합하고 중복을 제거합니다."""
        return list(dict.fromkeys(img for lst in image_lists for img in lst if img))

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


# 사용 예제
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 스크래퍼 초기화
    scraper = HaeoeumScraper(headless=True, output_dir="output")
    
    # 1. 단일 제품 이미지 추출 예제
    product_idx = "431692"  # 예시 상품 ID
    print(f"\n예제 1: 단일 제품 이미지 추출 (ID: {product_idx})")
    
    image_result = scraper.extract_images_for_excel_product(product_idx)
    print(f"이미지 URL: {image_result['image_url']}")
    print(f"이미지 갤러리: {image_result['image_gallery'][:3]}... (총 {len(image_result['image_gallery'])}개)")
    
    # 2. URL에서 제품 ID 추출 예제
    product_url = "http://www.jclgift.com/product/product_view.asp?p_idx=431692"
    print(f"\n예제 2: URL에서 제품 ID 추출")
    
    extracted_id = scraper._extract_product_idx(product_url)
    print(f"추출된 ID: {extracted_id}")
    
    # 3. 제품 URL에서 이미지 직접 추출 예제
    print(f"\n예제 3: 제품 URL에서 이미지 직접 추출")
    
    image_url = scraper.get_image_url_from_product_url(product_url)
    print(f"이미지 URL: {image_url}")
    
    # 4. 엑셀 데이터 예제
    print(f"\n예제 4: 엑셀 데이터에서 이미지 추출")
    
    # 샘플 엑셀 데이터
    excel_data = [
        {"상품명": "칼라인쇄 부동산화일 A4", "본사상품링크": "http://www.jclgift.com/product/product_view.asp?p_idx=431692"},
        {"상품명": "다른 상품", "본사상품링크": "http://www.jclgift.com/product/product_view.asp?p_idx=123456"}
    ]
    
    result = scraper.extract_images_from_excel_data(excel_data)
    
    for item in result:
        print(f"상품명: {item['상품명']}")
        print(f"이미지 URL: {item['image_url']}")
        print(f"상태: {item.get('image_status', 'Unknown')}")
        print()
