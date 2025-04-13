import asyncio
import configparser
import hashlib
import json
import logging
import os
import random
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Initialize absl logging
try:
    from absl import logging as absl_logging
    absl_logging.use_absl_handler()
    absl_logging.set_verbosity(absl_logging.INFO)
except ImportError:
    pass

# Try importing playwright (optional dependency)
try:
    from playwright.sync_api import TimeoutError, sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from utils.caching import FileCache, cache_result

from ..data_models import Product
from . import BaseMultiLayerScraper


class NaverShoppingAPI(BaseMultiLayerScraper):
    """
    네이버 쇼핑 API - 공식 API 활용 엔진

    특징:
    - 네이버 검색 API 활용
    - 비동기 작업 처리
    - 메모리 효율적 데이터 구조
    - 캐싱 지원
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        max_retries: int = 5,
        cache: Optional[FileCache] = None,
        timeout: int = 30,
    ):
        super().__init__(max_retries=max_retries, cache=cache, timeout=timeout)

        # 네이버 API 인증 정보
        self.client_id = client_id
        self.client_secret = client_secret

        # API 기본 URL 및 설정
        self.api_url = "https://openapi.naver.com/v1/search/shop.json"

        # 기본 헤더 설정
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Accept": "application/json",
        }

        # Define promotional site keywords (기존 코드에서 유지)
        self.promo_keywords = [
            "온오프마켓",
            "답례품",
            "기프트",
            "판촉",
            "기념품",
            "인쇄",
            "각인",
            "제작",
            "홍보",
            "미스터몽키",
            "호갱탈출",
            "고려기프트",
            "판촉물",
            "기업선물",
            "단체선물",
            "행사용품",
            "홍보물",
            "기업홍보",
            "로고인쇄",
            "로고각인",
            "로고제작",
            "기업답례품",
            "행사답례품",
            "기념품제작",
            "기업기념품",
        ]

        # Define promotional product categories (기존 코드에서 유지)
        self.promo_categories = [
            "기업선물",
            "단체선물",
            "행사용품",
            "홍보물",
            "기업홍보",
            "로고인쇄",
            "로고각인",
            "로고제작",
            "기업답례품",
            "행사답례품",
            "기념품제작",
            "기업기념품",
            "기업홍보물",
            "기업홍보물제작",
            "기업홍보물인쇄",
            "기업홍보물각인",
            "기업홍보물제작",
        ]

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Setup image comparison parameters
        self.require_image_match = (
            True  # 매뉴얼 요구사항: 이미지와 규격이 동일한 경우만 동일상품으로 판단
        )

        # Setup price filter
        self.min_price_diff_percent = 10  # 매뉴얼 요구사항: 10% 이하 가격차이 제품 제외

        self.max_pages = 3  # 매뉴얼 요구사항: 최대 3페이지까지만 탐색

        # 결과 개수 설정 (최대 100)
        self.display = 100

        # DNS 캐시 설정
        self.session = requests.Session()
        self.session.verify = False  # SSL 인증서 검증 비활성화
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=100,
            pool_maxsize=100
        ))
        self.session.mount('https://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=100,
            pool_maxsize=100
        ))

        # ThreadPoolExecutor 초기화
        self.executor = ThreadPoolExecutor(max_workers=10)  # 적절한 워커 수 설정

    def search_product(
        self, query: str, max_items: int = 50, reference_price: float = 0
    ) -> List[Product]:
        """
        네이버 쇼핑에서 제품 검색

        Args:
            query: 검색어
            max_items: 최대 검색 결과 수
            reference_price: 참조 가격 (10% 룰 적용용)

        Returns:
            List[Product]: 검색된 제품 목록
        """
        if self.cache:

            @cache_result(self.cache, key_prefix="naver_api_search")
            def cached_search(q, m, p):
                return self._search_product_logic(q, m, p)

            return cached_search(query, max_items, reference_price)
        else:
            return self._search_product_logic(query, max_items, reference_price)

    def _search_product_logic(
        self, query: str, max_items: int = 50, reference_price: float = 0
    ) -> List[Product]:
        """네이버 쇼핑 API 검색 핵심 로직"""
        # 비동기 함수를 동기적으로 실행
        return asyncio.run(
            self._search_product_async(query, max_items, reference_price)
        )

    async def _search_product_async(
        self, query: str, max_items: int = 50, reference_price: float = 0
    ) -> List[Product]:
        """비동기 방식으로 제품 검색 (병렬 처리)"""
        products = []

        try:
            self.logger.info(f"Searching Naver Shopping API for '{query}'")

            # 낮은 가격순 정렬 적용 (매뉴얼 요구사항)
            sort = "asc"  # 가격 오름차순

            # 최대 3페이지까지 검색 (매뉴얼 요구사항)
            for page in range(1, self.max_pages + 1):
                page_products = await self._fetch_api_results(query, page, sort)

                if not page_products:
                    break

                # Apply 10% rule if reference price is provided
                if reference_price > 0:
                    filtered_products = []
                    for product in page_products:
                        # Skip products with no price
                        if not product.price:
                            continue

                        # Calculate price difference percentage
                        price_diff_percent = (
                            (product.price - reference_price) / reference_price
                        ) * 100

                        # Include product if price difference is significant enough
                        # (either lower price or at least min_price_diff_percent higher)
                        if (
                            price_diff_percent < 0
                            or price_diff_percent >= self.min_price_diff_percent
                        ):
                            filtered_products.append(product)

                    page_products = filtered_products

                products.extend(page_products)

                # Stop if we have enough products or no more pages
                if len(products) >= max_items:
                    products = products[:max_items]
                    break

                # 페이지 간 지연 시간 (API 호출 제한 고려)
                wait_time = 0.5 + random.uniform(0.2, 0.5)
                self.logger.debug(
                    f"Waiting {wait_time:.2f} seconds before fetching next page"
                )
                await asyncio.sleep(wait_time)

            if not products:
                self.logger.info(
                    f"No products found for '{query}' on Naver Shopping API"
                )
                # 매뉴얼 요구사항: 찾지 못하면 "동일상품 없음"으로 처리
                no_match_product = Product(
                    id="no_match",
                    name=f"동일상품 없음 - {query}",
                    source="naver_shopping",
                    price=0,
                    url="",
                    image_url="",
                )
                products.append(no_match_product)
            else:
                self.logger.info(
                    f"Found {len(products)} products for '{query}' on Naver Shopping API"
                )
                # 여기서 no_match 제품을 추가하지 않음

            return products

        except Exception as e:
            self.logger.error(
                f"Error searching Naver Shopping API for '{query}': {str(e)}",
                exc_info=True,
            )
            return []

    async def _fetch_api_results(
        self, query: str, page: int, sort: str = "asc"
    ) -> List[Product]:
        """네이버 쇼핑 API 호출하여 결과 가져오기"""
        # Create cache key for this query and page
        cache_key = f"naver_api|{query}|{page}|{sort}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            return cached_result

        # API 요청 파라미터 설정
        params = {
            "query": query,
            "display": self.display,  # 한 페이지당 결과 수 (최대 100)
            "start": (page - 1) * self.display + 1,  # 페이지 시작점
            "sort": sort,  # 정렬 (asc: 가격 오름차순)
            "filter": "naverpay",  # 네이버페이 연동 상품만 (선택적)
            "exclude": "used:rental",  # 중고, 렌탈 제외
        }

        loop = asyncio.get_running_loop()
        retry_count = 0
        max_retries = self.max_retries
        retry_delay = 1.0

        while retry_count <= max_retries:
            try:
                # API 요청 실행
                response = await loop.run_in_executor(
                    self.executor,
                    lambda: requests.get(
                        self.api_url,
                        headers=self.headers,
                        params=params,
                        timeout=self.timeout,
                    ),
                )

                # 더 자세한 응답 로깅 추가
                self.logger.debug(f"API 요청: {self.api_url}")
                self.logger.debug(f"헤더: {self.headers}")
                self.logger.debug(f"파라미터: {params}")
                self.logger.info(f"응답 상태 코드: {response.status_code}")

                # 응답 내용 전체 로깅 (디버깅용)
                response_text = response.text
                self.logger.info(
                    f"응답 내용: {response_text[:200]}..."
                )  # 응답의 처음 200자만 로그에 출력

                # 응답 상태 코드 확인
                if response.status_code == 429:  # Too Many Requests
                    retry_count += 1
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(
                        f"Rate limit exceeded. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                elif response.status_code != 200:
                    self.logger.error(
                        f"API request failed with status code {response.status_code}: {response.text}"
                    )
                    return []

                # JSON 응답 파싱
                data = response.json()

                # 검색 결과가 없는 경우
                if data.get("total", 0) == 0 or not data.get("items"):
                    self.logger.info(f"No results found for '{query}' on page {page}")
                    return []

                # 결과 요약 로깅
                self.logger.info(
                    f"총 검색 결과: {data.get('total', 0)}개, 현재 페이지 아이템: {len(data.get('items', []))}개"
                )

                # 제품 데이터 변환
                products = []
                for item in data.get("items", []):
                    product = await self._convert_api_item_to_product(item)
                    if product:
                        products.append(product)

                # 캐시에 저장
                if products:
                    if self.cache:
                        self.cache_sparse_data(f"naver_api_{query}_{page}", products)

                return products

            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(
                        f"Error during API request: {str(e)}. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Failed to fetch API results after {max_retries} retries: {str(e)}"
                    )
                    return []

        return []

    async def _convert_api_item_to_product(
        self, item: Dict
    ) -> Optional[Product]:
        """API 응답 아이템을 Product 객체로 변환"""
        try:
            # HTML 태그 제거
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))

            # 가격 추출
            price = int(item.get("lprice", 0))

            # 제품 ID 생성
            product_id = (
                item.get("productId", "")
                or hashlib.md5(item.get("link", "").encode()).hexdigest()
            )

            # 카테고리 정보
            category = item.get("category1", "")
            if item.get("category2"):
                category += f" > {item.get('category2')}"
            if item.get("category3"):
                category += f" > {item.get('category3')}"
            if item.get("category4"):
                category += f" > {item.get('category4')}"

            # 판매처 정보
            mall_name = item.get("mallName", "")

            # 이미지 URL 처리
            image_url = item.get("image", "")
            
            # 이미지 URL 정규화
            if image_url:
                # HTTP를 HTTPS로 변환
                if image_url.startswith('http:'):
                    image_url = 'https:' + image_url[5:]
                # 프로토콜이 없는 경우 https: 추가
                elif image_url.startswith('//'):
                    image_url = 'https:' + image_url
            else:
                # 이미지가 없는 경우 기본 네이버 이미지 사용
                image_url = "https://ssl.pstatic.net/static/shop/front/techreview/web/resource/images/naver.png"
                self.logger.warning(f"상품 '{title}'의 이미지 URL이 없어 기본 이미지를 사용합니다.")

            # 홍보성 제품 여부 확인
            is_promotional = self._is_promotional_product(title, mall_name, category)

            # Product 객체 생성
            product = Product(
                id=product_id,
                name=title,
                price=price,
                source="naver_api",
                url=item.get("link", ""),
                image_url=image_url,
                brand=mall_name,
                category=category,
                is_promotional_site=is_promotional,
                original_input_data=item,
            )

            return product
        except Exception as e:
            self.logger.error(
                f"Error converting API item to Product: {str(e)}", exc_info=True
            )
            return None

    def _is_promotional_product(
        self, title: str, mall_name: str, category: str
    ) -> bool:
        """홍보성 제품 여부 확인 (기존 코드에서 유지)"""
        if not title:
            return False

        title_lower = title.lower()
        mall_name_lower = mall_name.lower() if mall_name else ""
        category_lower = category.lower() if category else ""

        # Check title against promotional keywords
        for keyword in self.promo_keywords:
            if keyword in title_lower or keyword in mall_name_lower:
                return True

        # Check category against promotional categories
        for cat in self.promo_categories:
            if cat in category_lower:
                return True

        return False

    def process_file(self, input_file: str) -> Tuple[Optional[str], Optional[str]]:
        """Process input Excel file and generate reports."""
        self.logger.error("This method is not fully implemented")
        return None, None


class NaverPriceTableCrawler:
    """
    네이버 쇼핑 및 연결된 쇼핑몰에서 가격표 추출

    특징:
    - Playwright 사용
    - 다양한 사이트의 가격표 추출 지원
    - 부가세 자동 계산
    - 수량별 가격 추출
    - 품절 여부 감지
    """

    def __init__(self, output_dir: str = "output", headless: bool = True):
        """
        가격표 크롤러 초기화

        Args:
            output_dir: 추출한 가격표를 저장할 디렉토리
            headless: 헤드리스 모드 여부 (브라우저 표시하지 않음)
        """
        self.output_dir = output_dir
        self.headless = headless
        self.logger = logging.getLogger(__name__)
        self.dialog_message = ""  # 대화 상자 메시지 저장

        # 스크래핑 실패 시 기본 반환값
        self.default_result = None, False, "Not available"

        if not PLAYWRIGHT_AVAILABLE:
            self.logger.warning(
                "Playwright is not installed. Price table extraction will not work."
            )

        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

    def _handle_dialog(self, dialog):
        """대화 상자 처리 (품절 등 상태 메시지 확인용)"""
        self.dialog_message = dialog.message
        self.logger.debug(f"Dialog message: {dialog.message}")
        dialog.accept()

    def _remove_special_chars(self, value):
        """문자와 특수 문자를 제거 (숫자만 추출)"""
        try:
            return "".join(filter(str.isdigit, str(value)))
        except (TypeError, ValueError):
            return value

    def _clean_quantity(self, qty):
        """수량 미만 처리 함수"""
        if isinstance(qty, str) and "미만" in qty:
            return "0"
        else:
            try:
                return "".join(filter(str.isdigit, str(qty)))
            except (TypeError, ValueError):
                return qty

    def get_price_table(self, url: str) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """URL에서 가격표 가져오기"""
        if not PLAYWRIGHT_AVAILABLE:
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

                # 네이버 쇼핑 링크인 경우, 최저가 사러가기 링크 추출
                if "naver.com" in url and "shopping" in url:
                    try:
                        lowest_price_link = page.locator(
                            '//div[contains(@class, "lowestPrice_btn_box")]/div[contains(@class, "buyButton_compare_wrap")]/a[text()="최저가 사러가기"]'
                        ).get_attribute("href")
                        if lowest_price_link:
                            self.logger.info(
                                "최저가 사러가기 링크 발견. 해당 페이지로 이동합니다."
                            )
                            page.goto(lowest_price_link, wait_until="networkidle")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to find or navigate to lowest price link: {e}"
                        )

                # 품절 확인
                is_sold_out = False
                if self.dialog_message and (
                    "상품" in self.dialog_message
                    or "재고" in self.dialog_message
                    or "품절" in self.dialog_message
                ):
                    is_sold_out = True
                    self.logger.info(f"상품 품절 확인: {self.dialog_message}")
                    browser.close()
                    return None, is_sold_out, self.dialog_message

                # 각 XPath와 연결된 함수 실행
                xpath_to_function = {
                    '//div[@class = "price-box"]': self._handle_login_one,  # 부가세 별도
                    '//div[@class = "tbl02"]': self._handle_login_one,  # 부가세 별도
                    '//table[@class = "hompy1004_table_class hompy1004_table_list"]/ancestor::td[1]': self._handle_login_two,  # 부가세 별도
                    '//table[@class = "goods_option"]//td[@colspan = "4"]': self._handle_login_three,  # 부가세 별도
                    '//div[@class = "vi_info"]//div[@class = "tbl_frm01"]': self._handle_login_one,  # 부가세 별도
                    '//div[@class = "specArea"]//div[@class = "w100"]': self._handle_login_one,
                }

                result_df = None

                # 각 XPath와 연결된 함수 실행
                for xpath, function in xpath_to_function.items():
                    element = page.query_selector(xpath)
                    if element:
                        html_content = element.inner_html()
                        soup = BeautifulSoup(html_content, "html.parser")
                        result_df = function(soup)  # soup을 함수로 전달

                        if result_df is not None:
                            self.logger.info(f"가격표 추출 성공: {len(result_df)} 행")
                            # CSV 파일로 저장
                            output_file = f"{self.output_dir}/unit_price_list.csv"
                            result_df.to_csv(output_file, index=False)
                            self.logger.info(f"가격표 저장 완료: {output_file}")
                            break

                browser.close()

                return result_df, is_sold_out, self.dialog_message

        except Exception as e:
            self.logger.error(f"Error in price table crawling: {e}", exc_info=True)
            return None, False, str(e)

    def _handle_login_one(self, soup):
        """첫 번째 유형의 가격표 처리"""
        try:
            # 테이블 찾기 시도
            tables = soup.find_all("table")
            if tables:
                df = pd.read_html(str(tables[0]))[
                    0
                ]  # 첫 번째 테이블을 DataFrame으로 변환
                df = df.T
                df.reset_index(drop=False, inplace=True)
                df.columns = df.iloc[0]
                df.drop(index=0, inplace=True)
                df.columns = ["수량", "일반"]
                df = df.map(self._remove_special_chars)
                df["일반"] = df["일반"].apply(lambda x: float(x) * 1.1)  # 부가세 추가
                df["수량"] = df["수량"].astype("int64")
                df.sort_values(by="수량", inplace=True, ignore_index=True)
                return df
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error in _handle_login_one: {e}")
            return None

    def _handle_login_two(self, soup):
        """두 번째 유형의 가격표 처리"""
        try:
            tables = soup.find_all("table")
            if tables:
                df = pd.read_html(str(tables[0]))[
                    0
                ]  # 첫 번째 테이블을 DataFrame으로 변환
                df = df.T
                df.reset_index(drop=False, inplace=True)
                df.columns = df.iloc[0]
                df.drop(index=0, inplace=True)
                df["수량"] = df["수량"].apply(self._clean_quantity)
                df = df.map(self._remove_special_chars)
                try:
                    df.drop("회원", axis=1, inplace=True)
                except Exception:
                    pass
                df.sort_values(by="수량", inplace=True, ignore_index=True)
                return df
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error in _handle_login_two: {e}")
            return None

    def _handle_login_three(self, soup):
        """세 번째 유형의 가격표 처리"""
        try:
            tables = soup.find_all("table")
            if tables:
                # 입력 태그에서 수량과 가격 추출
                quantities = []
                prices = []

                for input_tag in soup.find_all("input", class_="qu"):
                    try:
                        quantities.append(int(input_tag["value"]))
                    except (ValueError, KeyError):
                        pass

                for input_tag in soup.find_all("input", class_="pr"):
                    try:
                        prices.append(int(input_tag["value"].replace(",", "")))
                    except (ValueError, KeyError):
                        pass

                if quantities and prices and len(quantities) == len(prices):
                    # 데이터프레임 생성
                    df = pd.DataFrame({"수량": quantities, "일반": prices})
                    df["일반"] = df["일반"].apply(
                        lambda x: float(x) * 1.1
                    )  # 부가세 추가
                    df.sort_values(by="수량", inplace=True, ignore_index=True)
                    return df
            return None
        except Exception as e:
            self.logger.error(f"Error in _handle_login_three: {e}")
            return None

    def check_stock_status(self, url: str) -> Tuple[bool, str]:
        """상품 URL에서 재고 상태 확인"""
        _, is_sold_out, message = self.get_price_table(url)
        return not is_sold_out, message


class NaverShoppingCrawler(BaseMultiLayerScraper):
    """
    네이버 쇼핑 크롤러 - 공식 API와 연결하는 브릿지 클래스

    특징:
    - .env 파일 또는 설정에서 API 키 로드
    - NaverShoppingAPI 클래스의 래퍼
    - 재시도 및 예외 처리
    """

    def __init__(
        self,
        max_retries: int = 5,
        timeout: int = 30,
        cache: Optional[FileCache] = None,
    ):
        super().__init__(max_retries, timeout, cache)
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # API 키 로드
        self.api_keys = self._load_api_keys()

        # Naver Shopping API 인스턴스 생성
        self.api = NaverShoppingAPI(
            client_id=self.api_keys["client_id"],
            client_secret=self.api_keys["client_secret"],
            max_retries=max_retries,
            cache=cache,
            timeout=timeout,
        )

        # 단가표 크롤러 생성
        self.price_table_crawler = NaverPriceTableCrawler(
            output_dir="output",
            headless=False,  # 개발/디버깅 시 False, 배포 시 True로 변경
        )

    def _load_api_keys(self) -> Dict[str, str]:
        """
        config.ini 파일에서 네이버 API 키 로드
        config.ini에 없으면 .env 파일에서 찾음
        """
        import os
        from dotenv import load_dotenv

        # config.ini 파일에서 먼저 찾기
        try:
            config = configparser.ConfigParser()
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config.ini",
            )

            if os.path.exists(config_path):
                config.read(config_path, encoding="utf-8")

                if (
                    "API" in config
                    and "NAVER_CLIENT_ID" in config["API"]
                    and "NAVER_CLIENT_SECRET" in config["API"]
                ):
                    client_id = config["API"]["NAVER_CLIENT_ID"]
                    client_secret = config["API"]["NAVER_CLIENT_SECRET"]
                    self.logger.info("API 키를 config.ini 파일에서 로드했습니다.")
                    return {"client_id": client_id, "client_secret": client_secret}
        except Exception as e:
            self.logger.warning(
                f"config.ini 파일에서 API 키를 로드하는 중 오류 발생: {e}"
            )

        # config.ini에서 찾지 못하면 .env 파일에서 찾기
        load_dotenv()
        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")

        if not client_id or not client_secret:
            self.logger.error(
                "네이버 API 키를 찾을 수 없습니다. config.ini나 .env 파일에 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET이 설정되어 있는지 확인하세요."
            )
            raise ValueError("네이버 API 키를 찾을 수 없습니다.")

        self.logger.info(f"API 키를 .env 파일에서 로드했습니다. (client_id: {client_id[:4]}...)")
        return {"client_id": client_id, "client_secret": client_secret}

    def search_product(
        self, query: str, max_items: int = 50, reference_price: float = 0
    ) -> List[Product]:
        """
        네이버 쇼핑 API를 사용하여 제품 검색

        Args:
            query: 검색어
            max_items: 최대 검색 결과 수
            reference_price: 참조 가격 (10% 룰 적용용)

        Returns:
            List[Product]: 검색된 제품 목록
        """
        try:
            self.logger.info(f"네이버 쇼핑 API 검색 시작: '{query}'")

            # 검색어 전처리: 언더스코어를 공백으로 변환
            processed_query = query.replace("_", " ")
            if processed_query != query:
                self.logger.info(f"검색어 전처리: '{query}' -> '{processed_query}'")

            # NaverShoppingAPI를 통해 검색 수행
            products = self.api.search_product(
                query=processed_query,
                max_items=max_items,
                reference_price=reference_price,
            )

            # 검색 결과 검증
            if not products:
                self.logger.warning(f"'{processed_query}'에 대한 검색 결과가 없습니다.")
                # 결과가 없을 때 기본 제품 생성
                no_match_product = Product(
                    id="no_match",
                    name=f"동일상품 없음 - {processed_query}",
                    source="naver_shopping",
                    price=0,
                    url="",
                    image_url="https://ssl.pstatic.net/static/shop/front/techreview/web/resource/images/naver.png",
                )
                return [no_match_product]
            
            # 검색 결과 로그
            self.logger.info(f"'{processed_query}'에 대한 검색 결과 {len(products)}개 발견")
            
            # 이미지 URL 유효성 확인 및 수정
            valid_products = []
            for product in products:
                # 이미지 URL이 없는 경우 기본 이미지 설정
                if not product.image_url:
                    product.image_url = "https://ssl.pstatic.net/static/shop/front/techreview/web/resource/images/naver.png"
                    self.logger.warning(f"상품 '{product.name}'의 이미지 URL이 없어 기본 이미지를 사용합니다.")
                
                # 이미지 URL 표준화
                if product.image_url.startswith('http:'):
                    product.image_url = 'https:' + product.image_url[5:]
                elif product.image_url.startswith('//'):
                    product.image_url = 'https:' + product.image_url
                
                # 타임스탬프 추가
                product.fetched_at = datetime.now().isoformat()
                
                valid_products.append(product)
                self.logger.debug(f"상품 추가: {product.name} (이미지: {product.image_url[:50]}...)")
            
            return valid_products

        except Exception as e:
            self.logger.error(f"네이버 쇼핑 API 검색 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시에도 기본 제품 반환
            no_match_product = Product(
                id="no_match",
                name=f"검색 오류 - {query}",
                source="naver_shopping",
                price=0,
                url="",
                image_url="https://ssl.pstatic.net/static/shop/front/techreview/web/resource/images/naver.png",
            )
            return [no_match_product]

    def process_file(self, input_file: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Excel 파일 처리하고 보고서 생성

        Args:
            input_file: 처리할 입력 파일 경로

        Returns:
            Tuple[Optional[str], Optional[str]]: 생성된 보고서 파일 경로 및 로그 파일 경로
        """
        # 이 함수는 추후 구현이 필요함
        self.logger.error("process_file 메서드는 아직 구현되지 않았습니다.")
        return None, None

    def get_price_table(self, url: str) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """상품 URL에서 단가표 가져오기"""
        return self.price_table_crawler.get_price_table(url)

    def check_stock_status(self, url: str) -> Tuple[bool, str]:
        """상품 URL에서 재고 상태 확인"""
        return self.price_table_crawler.check_stock_status(url)
