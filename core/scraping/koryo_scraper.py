import re
import logging
import requests
import hashlib
from bs4 import BeautifulSoup
from time import sleep
from typing import List, Dict, Optional, Any, Tuple
import urllib.parse
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..data_models import Product
# Add imports for caching
from utils.caching import FileCache, cache_result
from . import BaseMultiLayerScraper, DOMExtractionStrategy, TextExtractionStrategy

class KoryoScraper(BaseMultiLayerScraper):
    """
    고려기프트 스크래퍼 - 다중 레이어 추출 엔진 활용
    
    특징:
    - DOM, 텍스트, 좌표 기반 추출 전략
    - 비동기 작업 처리
    - 메모리 효율적 데이터 구조
    - 선택적 요소 관찰
    """
    
    def __init__(self, max_retries: int = 3, cache: Optional[FileCache] = None, timeout: int = 30):
        super().__init__(max_retries=max_retries, cache=cache, timeout=timeout)
        
        # 사이트 관련 상수 정의
        self.base_url = "https://www.koreagift.co.kr"
        self.search_url = f"{self.base_url}/shop/search.php"
        
        # 요청 헤더
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": self.base_url
        }
        
        # 추출 셀렉터 정의 - 구조화된 다중 레이어 접근 방식
        self.selectors = {
            'product_list': {
                'selector': 'div.prd_wrap ul.prd_list li',
                'options': {'multiple': True}
            },
            'product_title': {
                'selector': 'div.prd_info p.name a',
                'options': {'multiple': False}
            },
            'product_link': {
                'selector': 'div.prd_info p.name a',
                'options': {'attribute': 'href'}
            },
            'price': {
                'selector': 'div.prd_info p.price',
                'options': {'multiple': False}
            },
            'thumbnail': {
                'selector': 'div.prd_img a img',
                'options': {'attribute': 'src'}
            },
            'next_page': {
                'selector': 'div.pagenation a.next',
                'options': {'attribute': 'href'}
            },
            # 상세 페이지 셀렉터
            'detail_title': {
                'selector': 'div.top_prd_name h1',
                'options': {'multiple': False}
            },
            'detail_price': {
                'selector': 'div.price_wrap p.price',
                'options': {'multiple': False}
            },
            'detail_code': {
                'selector': 'div.top_prd_name p.prd_code',
                'options': {'multiple': False}
            },
            'detail_images': {
                'selector': 'div.prd_img_wrap img',
                'options': {'multiple': True, 'attribute': 'src'}
            },
            'quantity_table': {
                'selector': 'table.price_tbl',
                'options': {'multiple': False}
            },
            'specs_table': {
                'selector': 'table.info_tbl',
                'options': {'multiple': False}
            },
            'description': {
                'selector': 'div.prd_detail',
                'options': {'multiple': False}
            }
        }
        
        # 텍스트 추출용 정규식 패턴
        self.patterns = {
            'price_number': re.compile(r'[\d,]+'),
            'product_code': re.compile(r'상품코드\s*:\s*([A-Za-z0-9-]+)'),
            'quantity': re.compile(r'(\d+)(개|세트|묶음)')
        }
    
    # 캐싱 적용 검색 메서드
    def search_product(self, query: str, max_items: int = 50) -> List[Product]:
        """캐싱을 적용한 제품 검색"""
        if self.cache:
            @cache_result(self.cache, key_prefix="koryo_search")
            def cached_search(q, m):
                 return self._search_product_logic(q, m)
            return cached_search(query, max_items)
        else:
            return self._search_product_logic(query, max_items)

    # 핵심 검색 로직
    def _search_product_logic(self, query: str, max_items: int = 50) -> List[Product]:
        """고려기프트 사이트에서 제품 검색"""
        # 비동기 함수를 동기적으로 실행
        return asyncio.run(self._search_product_async(query, max_items))
    
    async def _search_product_async(self, query: str, max_items: int = 50) -> List[Product]:
        """비동기 방식으로 제품 검색 (병렬 처리)"""
        products = []
        page = 1
        
        while True:
            try:
                self.logger.debug(f"Koryo Scraper: Searching for '{query}', page={page}")
                items = await self._crawl_page_async(query, page)
                
                if not items:
                    self.logger.warning(f"Koryo Scraper: No items found for '{query}' at page={page}.")
                    break
                
                # 병렬로 제품 상세 정보 가져오기
                detail_futures = []
                for item in items[:max_items - len(products) if max_items > 0 else len(items)]:
                    if 'link' in item:
                        detail_futures.append(self._get_product_details_async(item))
                
                if detail_futures:
                    detail_results = await asyncio.gather(*detail_futures, return_exceptions=True)
                    for result in detail_results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Error getting product details: {str(result)}")
                            continue
                        if result:
                            products.append(result)
                
                if max_items > 0 and len(products) >= max_items:
                    break  # 최대 항목 수 도달
                
                # 다음 페이지 확인
                has_next = await self._has_next_page_async(query, page)
                if not has_next:
                     break
                     
                page += 1
                await asyncio.sleep(1.0)  # 예의를 위한 딜레이
                
                # 최대 5페이지로 제한
                if page > 5:
                    self.logger.debug(f"Koryo Scraper: Reached maximum page limit (5)")
                    break
                
            except Exception as e:
                self.logger.error(f"Error during Koryo scraping for '{query}': {str(e)}", exc_info=True)
                break
        
        self.logger.info(f"Koryo Scraper: Found {len(products)} products for query '{query}'")
        return products[:max_items] if max_items > 0 else products
        
    async def _crawl_page_async(self, query: str, page: int = 1) -> List[Dict]:
        """비동기 방식으로 페이지 크롤링"""
        # 캐시 확인
        cache_key = f"koryo_page|{query}|{page}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            return cached_result
        
        # URL 인코딩 및 요청 준비
        encoded_query = urllib.parse.quote(query)
        url = f"{self.search_url}?q={encoded_query}&page={page}"
        
        # 비동기 HTTP 요청
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.get(url, headers=self.headers, timeout=self.timeout)
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching page {page} for query '{query}': HTTP {response.status_code}")
                return []
                
            # HTML 파싱
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 제품 목록 추출
            products = []
            product_elements = self.extract(soup, self.selectors['product_list']['selector'], 
                                           **self.selectors['product_list']['options'])
            
            if not product_elements:
                self.logger.warning(f"No product elements found on page {page} for query '{query}'")
                return []
                
            # 각 제품 정보 추출
            for element in product_elements:
                try:
                    product_data = await self._extract_list_item_async(element)
                    if product_data:
                        products.append(product_data)
                except Exception as e:
                    self.logger.warning(f"Error extracting product data: {str(e)}")
                    continue
            
            # 결과 캐싱
            if products:
                self.add_sparse_data(cache_key, products, ttl=3600)  # 1시간 캐싱
                
            return products
            
        except Exception as e:
            self.logger.error(f"Error crawling page {page} for query '{query}': {str(e)}", exc_info=True)
            return []
    
    async def _extract_list_item_async(self, element) -> Dict:
        """제품 목록 항목에서 기본 정보 추출"""
        try:
            # 다중 레이어 추출을 통한 데이터 추출
            title_element = self.extract(element, self.selectors['product_title']['selector'])
            title = title_element.text.strip() if title_element else ""
            
            link_element = self.extract(element, self.selectors['product_link']['selector'])
            link = link_element.get('href') if link_element else ""
            if link and not link.startswith('http'):
                link = f"{self.base_url}/{link.lstrip('/')}"
            
            price_element = self.extract(element, self.selectors['price']['selector'])
            price_text = price_element.text.strip() if price_element else "0"
            price_match = self.patterns['price_number'].search(price_text)
            price = int(price_match.group().replace(',', '')) if price_match else 0
            
            thumbnail_element = self.extract(element, self.selectors['thumbnail']['selector'])
            thumbnail = thumbnail_element.get('src') if thumbnail_element else ""
            if thumbnail and not thumbnail.startswith('http'):
                thumbnail = f"{self.base_url}/{thumbnail.lstrip('/')}"
            
            # 고유 ID 생성
            product_id = hashlib.md5(link.encode()).hexdigest() if link else ""
            
            return {
                'title': title,
                'link': link,
                'price': price,
                'image': thumbnail,
                'product_id': product_id
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting list item: {str(e)}")
            return {}
            
    async def _get_product_details_async(self, item: Dict) -> Optional[Product]:
        """비동기 방식으로 제품 상세 정보 가져오기"""
        if not item.get('link'):
            return None
            
        # 캐시 확인
        cache_key = f"koryo_detail|{item['product_id']}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            return cached_result
            
        url = item['link']
        loop = asyncio.get_running_loop()
        
        try:
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.get(url, headers=self.headers, timeout=self.timeout)
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching product details: HTTP {response.status_code}")
                return None
                
            # HTML 파싱
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 상세 정보 추출
            detail_title = self.extract(soup, self.selectors['detail_title']['selector'])
            title = detail_title.text.strip() if detail_title else item.get('title', '')
            
            detail_price_element = self.extract(soup, self.selectors['detail_price']['selector'])
            price_text = detail_price_element.text.strip() if detail_price_element else "0"
            price_match = self.patterns['price_number'].search(price_text)
            price = int(price_match.group().replace(',', '')) if price_match else item.get('price', 0)
            
            # 제품 코드 추출
            code_element = self.extract(soup, self.selectors['detail_code']['selector'])
            product_code = ""
            if code_element:
                code_text = code_element.text.strip()
                code_match = self.patterns['product_code'].search(code_text)
                product_code = code_match.group(1) if code_match else ""
            
            # 이미지 URL 추출
            image_elements = self.extract(soup, self.selectors['detail_images']['selector'], 
                                         multiple=True, attribute='src')
            image_gallery = []
            if isinstance(image_elements, list):
                for img in image_elements:
                    img_url = img
                    if img_url and not img_url.startswith('http'):
                        img_url = f"{self.base_url}/{img_url.lstrip('/')}"
                    if img_url:
                        image_gallery.append(img_url)
            
            # 수량별 가격 추출
            quantity_prices = {}
            quantity_table = self.extract(soup, self.selectors['quantity_table']['selector'])
            if quantity_table:
                quantity_prices = await self._extract_quantity_prices_async(quantity_table)
            
            # 제품 사양 추출
            specifications = {}
            specs_table = self.extract(soup, self.selectors['specs_table']['selector'])
            if specs_table:
                specifications = self._extract_specifications(specs_table)
            
            # 제품 설명 추출
            description = ""
            desc_element = self.extract(soup, self.selectors['description']['selector'])
            if desc_element:
                description = desc_element.text.strip()
            
            # 제품 객체 생성
            product = Product(
                id=item.get('product_id', ''),
                name=title,
                price=price,
                source='koryo',
                original_input_data=item
            )
            
            # 추가 정보 설정
            product.url = url
            product.image_url = item.get('image', '') or (image_gallery[0] if image_gallery else '')
            product.image_gallery = image_gallery
            product.product_code = product_code
            product.description = description
            product.specifications = specifications
            product.quantity_prices = quantity_prices
            
            # 캐시에 저장
            self.add_sparse_data(cache_key, product, ttl=86400)  # 24시간 캐싱
            
            return product
            
        except Exception as e:
            self.logger.error(f"Error getting product details for {url}: {str(e)}", exc_info=True)
            return None

    async def _extract_quantity_prices_async(self, table_element) -> Dict[str, float]:
        """수량별 가격 테이블에서 가격 정보 추출"""
        quantity_prices = {}
        try:
            # 비동기 컨텍스트에서 실행하기 위해 쓰레드풀에서 실행
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor,
                self._extract_quantity_prices_sync,
                table_element
            )
        except Exception as e:
            self.logger.error(f"Error extracting quantity prices: {str(e)}")
        
        return quantity_prices
    
    def _extract_quantity_prices_sync(self, table_element) -> Dict[str, float]:
        """수량별 가격 테이블에서 가격 정보 추출 (동기 버전)"""
        quantity_prices = {}
        try:
            rows = table_element.select('tr')
            for row in rows[1:]:  # 헤더 행 제외
                cells = row.select('td')
                if len(cells) >= 2:
                    qty_cell = cells[0].text.strip()
                    price_cell = cells[1].text.strip()
                    
                    # 수량 추출
                    qty_match = self.patterns['quantity'].search(qty_cell)
                    qty = int(qty_match.group(1)) if qty_match else 0
                    
                    # 가격 추출
                    price_match = self.patterns['price_number'].search(price_cell)
                    price = int(price_match.group().replace(',', '')) if price_match else 0
                    
                    if qty and price:
                        quantity_prices[str(qty)] = price
        except Exception as e:
            self.logger.error(f"Error in _extract_quantity_prices_sync: {str(e)}")
        
        return quantity_prices
    
    def _extract_specifications(self, table_element) -> Dict[str, str]:
        """제품 사양 테이블에서 정보 추출"""
        specs = {}
        try:
            rows = table_element.select('tr')
            for row in rows:
                header_cell = row.select_one('th')
                value_cell = row.select_one('td')
                
                if header_cell and value_cell:
                    key = header_cell.text.strip()
                    value = value_cell.text.strip()
                    if key and value:
                        specs[key] = value
        except Exception as e:
            self.logger.error(f"Error extracting specifications: {str(e)}")
        
        return specs
    
    async def _has_next_page_async(self, query: str, current_page: int) -> bool:
        """다음 페이지 존재 여부 확인"""
        encoded_query = urllib.parse.quote(query)
        next_page = current_page + 1
        url = f"{self.search_url}?q={encoded_query}&page={next_page}"
        
        loop = asyncio.get_running_loop()
        try:
            # HEAD 요청으로 가볍게 확인
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.head(url, headers=self.headers, timeout=self.timeout)
            )
            
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Error checking next page: {str(e)}")
            return False 