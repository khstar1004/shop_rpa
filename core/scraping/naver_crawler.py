from time import sleep
import re
import logging
import requests
import hashlib
from typing import List, Dict, Optional, Any, Union, Tuple
from bs4 import BeautifulSoup
import urllib.parse
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import random

from ..data_models import Product
from utils.caching import FileCache, cache_result
from . import BaseMultiLayerScraper, DOMExtractionStrategy, TextExtractionStrategy

# 무료 프록시 서비스 목록 (필요에 따라 업데이트 필요)
FREE_PROXIES = [
    # 실제 프로젝트에서는 유료 프록시 서비스나 프록시 풀을 사용하는 것이 좋습니다
    # 아래는 예시일 뿐이므로 실제 작동하는 프록시로 교체해야 합니다
    # "http://1.2.3.4:8080",
    # "http://5.6.7.8:8080"
]

class NaverShoppingCrawler(BaseMultiLayerScraper):
    """
    네이버 쇼핑 크롤러 - 다중 레이어 추출 엔진 활용
    
    특징:
    - DOM, 텍스트, 좌표 기반 추출 전략
    - 비동기 작업 처리
    - 메모리 효율적 데이터 구조
    - 선택적 요소 관찰
    """
    
    def __init__(self, max_retries: int = 3, cache: Optional[FileCache] = None, timeout: int = 30, use_proxies: bool = False):
        super().__init__(max_retries=max_retries, cache=cache, timeout=timeout)
        
        # 프록시 사용 설정
        self.use_proxies = use_proxies
        self.proxies = FREE_PROXIES
        self.current_proxy_index = 0
        
        # Define promotional site keywords
        self.promo_keywords = [
            "온오프마켓", "답례품", "기프트", "판촉", "기념품", 
            "인쇄", "각인", "제작", "홍보", "미스터몽키", "호갱탈출",
            "고려기프트", "판촉물", "기업선물", "단체선물", "행사용품",
            "홍보물", "기업홍보", "로고인쇄", "로고각인", "로고제작",
            "기업답례품", "행사답례품", "기념품제작", "기업기념품"
        ]
        
        # Define promotional product categories
        self.promo_categories = [
            "기업선물", "단체선물", "행사용품", "홍보물", "기업홍보",
            "로고인쇄", "로고각인", "로고제작", "기업답례품", "행사답례품",
            "기념품제작", "기업기념품", "기업홍보물", "기업홍보물제작",
            "기업홍보물인쇄", "기업홍보물각인", "기업홍보물제작"
        ]
        
        self.base_url = "https://search.shopping.naver.com/search/all"
        
        # 다양한 사용자 에이전트를 추가하여 차단 방지
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        ]
        
        # 기본 헤더 설정
        self.headers = {
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://search.shopping.naver.com/",
            "sec-ch-ua": '"Not A(Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"',
            "sec-ch-ua-mobile": "?0",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Define extraction selectors - structured for multi-layer approach
        self.selectors = {
            'product_list': {
                'selector': 'div.basicList_item_inner__eY_mq',
                'options': {'multiple': True}
            },
            'product_title': {
                'selector': 'div.basicList_title__3P9Q7 a',
                'options': {'multiple': False}
            },
            'product_link': {
                'selector': 'div.basicList_title__3P9Q7 a',
                'options': {'attribute': 'href'}
            },
            'mall_name': {
                'selector': 'div.basicList_mall__1bQqQ a',
                'options': {'multiple': False}
            },
            'price': {
                'selector': 'div.basicList_price_box__1Hr7h span.price_num__2WUXn',
                'options': {'multiple': False}
            },
            'price_unit': {
                'selector': 'span.price_unit__BWL2W',
                'options': {'multiple': False}
            },
            'next_page': {
                'selector': 'a.pagination_next__3msKY',
                'options': {'attribute': 'href'}
            },
            'related_keywords': {
                'selector': 'div.relatedKeyword_related_keyword__Hph8J li button span.relatedKeyword_text__47X1f',
                'options': {'multiple': True}
            }
        }
        
        # 텍스트 추출용 정규식 패턴
        self.patterns = {
            'json_data': re.compile(r'window\.__PRELOADED_STATE__\s*=\s*({.*?});', re.DOTALL),
            'price_number': re.compile(r'[\d,]+'),
            'quantity': re.compile(r'(\d+)(개|세트|묶음)')
        }
        
        # 추출 전략 커스터마이징
        self._customize_extraction_strategies()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup image comparison parameters
        self.require_image_match = True  # 매뉴얼 요구사항: 이미지와 규격이 동일한 경우만 동일상품으로 판단
        
        # Setup price filter
        self.min_price_diff_percent = 10  # 매뉴얼 요구사항: 10% 이하 가격차이 제품 제외
        
        self.max_pages = 3  # 매뉴얼 요구사항: 최대 3페이지까지만 탐색
    
    def _customize_extraction_strategies(self):
        """특화된 추출 전략으로 기본 전략 확장"""
        # JSON 데이터 추출 전략 추가 (네이버 쇼핑 특화)
        class NaverJsonExtractionStrategy(TextExtractionStrategy):
            def extract(self, source, selector, **kwargs):
                if isinstance(selector, re.Pattern):
                    matches = selector.findall(source)
                    if matches:
                        try:
                            json_str = matches[0]
                            json_data = json.loads(json_str)
                            
                            # 특정 데이터 접근 경로가 있는 경우
                            if 'path' in kwargs:
                                path = kwargs['path'].split('.')
                                for key in path:
                                    if key.isdigit():
                                        json_data = json_data[int(key)]
                                    else:
                                        json_data = json_data.get(key, {})
                            return json_data
                        except (json.JSONDecodeError, IndexError, KeyError) as e:
                            logging.debug(f"JSON extraction failed: {str(e)}")
                            return None
                return super().extract(source, selector, **kwargs)
        
        # 전략 추가
        self.extraction_strategies.insert(0, NaverJsonExtractionStrategy())
    
    def search_product(self, query: str, max_items: int = 50, reference_price: float = 0) -> List[Product]:
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
            @cache_result(self.cache, key_prefix="naver_search")
            def cached_search(q, m, p):
                return self._search_product_logic(q, m, p)
            return cached_search(query, max_items, reference_price)
        else:
            return self._search_product_logic(query, max_items, reference_price)
    
    def _search_product_logic(self, query: str, max_items: int = 50, reference_price: float = 0) -> List[Product]:
        """네이버 쇼핑 검색 핵심 로직"""
        # 비동기 함수를 동기적으로 실행
        return asyncio.run(self._search_product_async(query, max_items, reference_price))
    
    async def _search_product_async(self, query: str, max_items: int = 50, reference_price: float = 0) -> List[Product]:
        """비동기 방식으로 제품 검색 (병렬 처리)"""
        products = []
        
        try:
            self.logger.info(f"Searching Naver Shopping for '{query}'")
            
            # 낮은 가격순 정렬 적용 (매뉴얼 요구사항)
            sort_param = "price_asc"
            
            # 최대 3페이지까지 검색 (매뉴얼 요구사항)
            for page in range(1, self.max_pages + 1):
                page_products = await self._crawl_page_async(query, page, sort_param)
                
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
                        price_diff_percent = ((product.price - reference_price) / reference_price) * 100
                        
                        # Include product if price difference is significant enough
                        # (either lower price or at least min_price_diff_percent higher)
                        if price_diff_percent < 0 or price_diff_percent >= self.min_price_diff_percent:
                            filtered_products.append(product)
                    
                    page_products = filtered_products
                
                products.extend(page_products)
                
                # Stop if we have enough products or no more pages
                if len(products) >= max_items:
                    products = products[:max_items]
                    break
                
                # 페이지 간 지연 시간 증가 (차단 방지)
                wait_time = 3.0 + random.uniform(2.0, 5.0)
                self.logger.debug(f"Waiting {wait_time:.2f} seconds before fetching next page")
                await asyncio.sleep(wait_time)
            
            if not products:
                self.logger.info(f"No products found for '{query}' on Naver Shopping")
            else:
                self.logger.info(f"Found {len(products)} products for '{query}' on Naver Shopping")
            
            # 매뉴얼 요구사항: 찾지 못하면 "동일상품 없음"으로 처리
            if not products:
                # Create a dummy product to indicate "no match found"
                no_match_product = Product(
                    id="no_match",
                    name=f"동일상품 없음 - {query}",
                    source="naver_shopping",
                    price=0,
                    url="",
                    image_url=""
                )
                products.append(no_match_product)
                
            return products
            
        except Exception as e:
            self.logger.error(f"Error searching Naver Shopping for '{query}': {str(e)}", exc_info=True)
            return []
    
    async def _crawl_page_async(self, query: str, page: int, sort: str = "price_asc") -> List[Product]:
        """비동기 방식으로 페이지 크롤링"""
        # Create cache key for this query and page
        cache_key = f"naver_page|{query}|{page}|{sort}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            return cached_result
        
        # Encode query for URL
        encoded_query = urllib.parse.quote(query)
        url = f"{self.base_url}?query={encoded_query}&pagingIndex={page}&sort={sort}"
        
        self.logger.debug(f"Crawling Naver Shopping page {page} for '{query}'")
        
        # Make async HTTP request
        loop = asyncio.get_running_loop()
        retry_count = 0
        max_retries = self.max_retries
        retry_delay = 2.0  # 초기 재시도 지연 시간
        
        while retry_count <= max_retries:
            try:
                # 매 요청마다 다른 사용자 에이전트 사용
                current_headers = self.headers.copy()
                current_headers["User-Agent"] = random.choice(self.user_agents)
                
                # 프록시 설정 (사용하는 경우)
                proxy = None
                if self.use_proxies and self.proxies:
                    # 순차적으로 프록시 변경
                    proxy = self.proxies[self.current_proxy_index % len(self.proxies)]
                    self.current_proxy_index += 1
                    self.logger.debug(f"Using proxy: {proxy}")
                
                # requests는 동기식이므로 ThreadPoolExecutor를 사용해 비동기로 변환
                response = await loop.run_in_executor(
                    self.executor,
                    lambda: requests.get(
                        url, 
                        headers=current_headers, 
                        timeout=self.timeout,
                        proxies={"http": proxy, "https": proxy} if proxy else None
                    )
                )
                
                # 418 또는 기타 오류 상태 코드 처리
                if response.status_code == 418:
                    retry_count += 1
                    if retry_count <= max_retries:
                        # 지수 백오프로 대기 시간 증가 (차단 우회를 위해)
                        wait_time = retry_delay * (2 ** (retry_count - 1)) + random.uniform(1, 5)
                        self.logger.warning(f"Received 418 status code. Retrying in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        self.logger.error(f"Error fetching page {page} for query '{query}': HTTP 418 (I'm a teapot) - Bot detection triggered")
                        return []
                
                elif response.status_code != 200:
                    self.logger.error(f"Error fetching page {page} for query '{query}': HTTP {response.status_code}")
                    return []
                
                # 요청 성공 처리
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    # 지수 백오프로 대기 시간 증가
                    wait_time = retry_delay * (2 ** (retry_count - 1)) + random.uniform(1, 3)
                    self.logger.warning(f"Error during request: {str(e)}. Retrying in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Error crawling Naver Shopping page {page} for '{query}' after {max_retries} retries: {str(e)}")
                    return []

        # 다중 추출 전략 활용
        # 1. JSON 데이터 추출 시도 (최신 네이버 쇼핑은 JSON 데이터를 사용)
        json_data = self.extract(response.text, self.patterns['json_data'], path='props.pageProps.initialState.products.list')
        
        if json_data and isinstance(json_data, list):
            # JSON 데이터에서 제품 추출
            items = []
            for item in json_data:
                try:
                    product_data = self._extract_from_json(item)
                    if product_data:
                        items.append(product_data)
                except Exception as e:
                    self.logger.warning(f"Error parsing JSON product: {str(e)}")
                    continue
                
            # 캐시에 저장
            if items:
                self.add_sparse_data(cache_key, items, ttl=3600)  # 1시간 캐싱
            
            return items
        
        # 2. 실패하면 전통적인 HTML 파싱
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 다중 레이어 추출 활용
        product_elements = self.extract(soup, self.selectors['product_list']['selector'], 
                                       **self.selectors['product_list']['options'])
        
        if not product_elements:
            self.logger.warning(f"No product elements found on page {page} for query '{query}'")
            return []
        
        items = []
        for element in product_elements:
            try:
                product_data = self._extract_product_data(element)
                if product_data:
                    items.append(product_data)
            except Exception as e:
                self.logger.warning(f"Error parsing product element: {str(e)}")
                continue
        
        # 캐시에 저장
        if items:
            self.add_sparse_data(cache_key, items, ttl=3600)  # 1시간 캐싱
        
        return items
    
    def _extract_from_json(self, item: Dict) -> Dict:
        """JSON 데이터에서 제품 정보 추출"""
        try:
            product_data = {
                'title': item.get('productTitle', ''),
                'link': item.get('mallProductUrl', ''),
                'image': item.get('imageUrl', ''),
                'price': item.get('price', {}).get('value', 0),
                'mall_name': item.get('mallName', ''),
                'category': item.get('category1Name', ''),
                'product_id': item.get('id', ''),
                'is_promotional': False
            }
            
            # 홍보성 제품 여부 검사
            product_data['is_promotional'] = self._is_promotional_product(
                product_data['title'], 
                product_data['mall_name'],
                product_data['category']
            )
            
            return product_data
        except Exception as e:
            self.logger.warning(f"Error extracting JSON product data: {str(e)}")
            return {}
    
    def _extract_product_data(self, element) -> Dict:
        """HTML 요소에서 제품 정보 추출"""
        try:
            # 다중 레이어 추출을 통한 제품 정보 추출
            title_element = self.extract(element, self.selectors['product_title']['selector'])
            title = title_element.text.strip() if title_element else ""
            
            link_element = self.extract(element, self.selectors['product_link']['selector'])
            link = link_element.get('href') if link_element else ""
            
            price_element = self.extract(element, self.selectors['price']['selector'])
            price_text = price_element.text.strip() if price_element else "0"
            price = int(re.sub(r'[^\d]', '', price_text)) if price_text else 0
            
            mall_element = self.extract(element, self.selectors['mall_name']['selector'])
            mall_name = mall_element.text.strip() if mall_element else ""
            
            # 이미지 URL은 다양한 속성에 저장될 수 있음
            image = ""
            img_element = element.select_one('img')
            if img_element:
                image = img_element.get('src', '')
                if not image:
                    image = img_element.get('data-src', '')
            
            # 카테고리 정보
            category = ""
            category_element = element.select_one('div.basicList_depth__2QIie')
            if category_element:
                category = category_element.text.strip()
            
            product_data = {
                'title': title,
                'link': link,
                'image': image,
                'price': price,
                'mall_name': mall_name,
                'category': category,
                'product_id': hashlib.md5(link.encode()).hexdigest(),
                'is_promotional': False
            }
            
            # 홍보성 제품 여부 검사
            product_data['is_promotional'] = self._is_promotional_product(title, mall_name, category)
            
            return product_data
        except Exception as e:
            self.logger.warning(f"Error extracting product data: {str(e)}")
            return {}

    def _is_promotional_product(self, title: str, mall_name: str, category: str) -> bool:
        """홍보성 제품 여부 확인"""
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

    async def _get_product_details_async(self, item: Dict) -> Optional[Product]:
        """비동기 방식으로 제품 상세 정보 가져오기"""
        # 기본 제품 객체 생성
        try:
            product = Product(
                id=item.get('product_id', ''),
                name=item.get('title', ''),
                price=item.get('price', 0),
                source='naver',
                original_input_data=item
            )
            
            # URL 및 이미지 URL 설정
            product.url = item.get('link', '')
            product.image_url = item.get('image', '')
            
            # 추가 정보 설정
            product.brand = item.get('mall_name', '')
            product.category = item.get('category', '')
            product.is_promotional_site = item.get('is_promotional', False)
            
            # 가격 정보가 없는 경우 0으로 설정
            if product.price == 0 and 'price' in item:
                try:
                    price_str = str(item['price'])
                    price_digits = re.sub(r'[^\d]', '', price_str)
                    if price_digits:
                        product.price = int(price_digits)
                except (ValueError, TypeError):
                    pass
            
            return product
        except Exception as e:
            self.logger.error(f"Error creating Product object: {str(e)}", exc_info=True)
            return None

    async def _check_next_page_async(self, query: str, next_page: int) -> bool:
        """비동기 방식으로 다음 페이지 존재 여부 확인"""
        encoded_query = urllib.parse.quote(query)
        url = f"{self.base_url}?query={encoded_query}&pagingIndex={next_page}"
        
        loop = asyncio.get_running_loop()
        try:
            # 가벼운 헤드 요청으로 확인
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.head(url, headers=self.headers, timeout=self.timeout)
            )
            
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Error checking next page: {str(e)}")
            return False 

    def process_file(self, input_file: str) -> Tuple[Optional[str], Optional[str]]:
        """Process input Excel file and generate reports."""
        self.logger.error("This method is not fully implemented")
        return None, None 