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

from ..data_models import Product
from utils.caching import FileCache, cache_result
from . import BaseMultiLayerScraper, DOMExtractionStrategy, TextExtractionStrategy

class NaverShoppingCrawler(BaseMultiLayerScraper):
    """
    네이버 쇼핑 크롤러 - 다중 레이어 추출 엔진 활용
    
    특징:
    - DOM, 텍스트, 좌표 기반 추출 전략
    - 비동기 작업 처리
    - 메모리 효율적 데이터 구조
    - 선택적 요소 관찰
    """
    
    def __init__(self, max_retries: int = 3, cache: Optional[FileCache] = None, timeout: int = 30):
        super().__init__(max_retries=max_retries, cache=cache, timeout=timeout)
        
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
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
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
    
    # Add caching decorator if cache is available
    def search_product(self, query: str, max_items: int = 50) -> List[Product]:
        """캐싱을 적용한 제품 검색"""
        if self.cache:
             @cache_result(self.cache, key_prefix="naver_search")
             def cached_search(q, m):
                 return self._search_product_logic(q, m)
             return cached_search(query, max_items)
        else:
             return self._search_product_logic(query, max_items)

    async def _search_product_async(self, query: str, max_items: int = 50) -> List[Product]:
        """비동기 방식으로 제품 검색 (병렬 처리)"""
        products = []
        page = 1
        total_processed = 0

        while True:
            try:
                self.logger.debug(f"Naver Crawler: Searching for '{query}', page={page}")
                items = await self._crawl_page_async(query, page)
                
                if not items:
                    self.logger.warning(f"Naver Crawler: No items found for '{query}' at page={page}.")
                    break
                
                # Process items
                # 병렬로 제품 상세 정보 가져오기
                tasks = []
                for item in items[:max_items - len(products) if max_items > 0 else len(items)]:
                    if 'link' in item:
                        tasks.append(self._get_product_details_async(item))
                    
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Error getting product details: {str(result)}")
                            continue
                        if result:
                            products.append(result)
                        if max_items > 0 and len(products) >= max_items:
                            break
                
                if max_items > 0 and len(products) >= max_items:
                     break  # Reached max_items limit
                
                # Update page and total processed
                page += 1
                total_processed += len(items)

                # 다음 페이지 존재 확인
                has_next = await self._check_next_page_async(query, page)
                if not has_next:
                        self.logger.debug(f"Naver Crawler: No more pages available for '{query}'")
                        break
                
                # Politeness delay
                await asyncio.sleep(1.5)  # Longer delay for crawling to avoid IP blocking
                
                # Limit to 5 pages maximum
                if page > 5:
                    self.logger.debug(f"Naver Crawler: Reached maximum page limit (5)")
                    break
                
            except Exception as e:
                self.logger.error(f"Error during Naver crawling for '{query}': {str(e)}", exc_info=True)
                break
        
        self.logger.info(f"Naver Crawler: Found {len(products)} products for query '{query}'")
        return products[:max_items] if max_items > 0 else products
    
    # Core crawling logic
    def _search_product_logic(self, query: str, max_items: int = 50) -> List[Product]:
        """Core logic to search for products using Naver Shopping crawler"""
        # 비동기 함수를 동기적으로 실행
        return asyncio.run(self._search_product_async(query, max_items))
    
    async def _crawl_page_async(self, query: str, page: int = 1) -> List[Dict]:
        """비동기 방식으로 페이지 크롤링"""
        # 이미 캐시에 있는지 확인
        cache_key = f"naver_page|{query}|{page}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            return cached_result
        
        # URL 인코딩
        encoded_query = urllib.parse.quote(query)
        url = f"{self.base_url}?query={encoded_query}&pagingIndex={page}"
        
        # 비동기로 HTTP 요청
        loop = asyncio.get_running_loop()
        try:
            # requests는 동기식이므로 ThreadPoolExecutor를 사용해 비동기로 변환
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.get(url, headers=self.headers, timeout=self.timeout)
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching page {page} for query '{query}': HTTP {response.status_code}")
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
            
        except Exception as e:
            self.logger.error(f"Error crawling page {page} for query '{query}': {str(e)}", exc_info=True)
            return []
    
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