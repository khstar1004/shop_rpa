from time import sleep
import re
import logging
import requests
import hashlib
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import random
import urllib.parse
import configparser

from ..data_models import Product
from utils.caching import FileCache, cache_result
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
    
    def __init__(self, client_id: str, client_secret: str, max_retries: int = 3, 
                 cache: Optional[FileCache] = None, timeout: int = 30):
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
            "Accept": "application/json"
        }
        
        # Define promotional site keywords (기존 코드에서 유지)
        self.promo_keywords = [
            "온오프마켓", "답례품", "기프트", "판촉", "기념품", 
            "인쇄", "각인", "제작", "홍보", "미스터몽키", "호갱탈출",
            "고려기프트", "판촉물", "기업선물", "단체선물", "행사용품",
            "홍보물", "기업홍보", "로고인쇄", "로고각인", "로고제작",
            "기업답례품", "행사답례품", "기념품제작", "기업기념품"
        ]
        
        # Define promotional product categories (기존 코드에서 유지)
        self.promo_categories = [
            "기업선물", "단체선물", "행사용품", "홍보물", "기업홍보",
            "로고인쇄", "로고각인", "로고제작", "기업답례품", "행사답례품",
            "기념품제작", "기업기념품", "기업홍보물", "기업홍보물제작",
            "기업홍보물인쇄", "기업홍보물각인", "기업홍보물제작"
        ]
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup image comparison parameters
        self.require_image_match = True  # 매뉴얼 요구사항: 이미지와 규격이 동일한 경우만 동일상품으로 판단
        
        # Setup price filter
        self.min_price_diff_percent = 10  # 매뉴얼 요구사항: 10% 이하 가격차이 제품 제외
        
        self.max_pages = 3  # 매뉴얼 요구사항: 최대 3페이지까지만 탐색
        
        # 결과 개수 설정 (최대 100)
        self.display = 100
    
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
            @cache_result(self.cache, key_prefix="naver_api_search")
            def cached_search(q, m, p):
                return self._search_product_logic(q, m, p)
            return cached_search(query, max_items, reference_price)
        else:
            return self._search_product_logic(query, max_items, reference_price)
    
    def _search_product_logic(self, query: str, max_items: int = 50, reference_price: float = 0) -> List[Product]:
        """네이버 쇼핑 API 검색 핵심 로직"""
        # 비동기 함수를 동기적으로 실행
        return asyncio.run(self._search_product_async(query, max_items, reference_price))
    
    async def _search_product_async(self, query: str, max_items: int = 50, reference_price: float = 0) -> List[Product]:
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
                
                # 페이지 간 지연 시간 (API 호출 제한 고려)
                wait_time = 0.5 + random.uniform(0.2, 0.5)
                self.logger.debug(f"Waiting {wait_time:.2f} seconds before fetching next page")
                await asyncio.sleep(wait_time)
            
            if not products:
                self.logger.info(f"No products found for '{query}' on Naver Shopping API")
            else:
                self.logger.info(f"Found {len(products)} products for '{query}' on Naver Shopping API")
            
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
            self.logger.error(f"Error searching Naver Shopping API for '{query}': {str(e)}", exc_info=True)
            return []
    
    async def _fetch_api_results(self, query: str, page: int, sort: str = "asc") -> List[Product]:
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
            "exclude": "used:rental"  # 중고, 렌탈 제외
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
                        timeout=self.timeout
                    )
                )
                
                # 더 자세한 응답 로깅 추가
                self.logger.debug(f"API 요청: {self.api_url}")
                self.logger.debug(f"헤더: {self.headers}")
                self.logger.debug(f"파라미터: {params}")
                self.logger.info(f"응답 상태 코드: {response.status_code}")
                
                # 응답 내용 전체 로깅 (디버깅용)
                response_text = response.text
                self.logger.info(f"응답 내용: {response_text[:200]}...")  # 응답의 처음 200자만 로그에 출력
                
                # 응답 상태 코드 확인
                if response.status_code == 429:  # Too Many Requests
                    retry_count += 1
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                
                elif response.status_code != 200:
                    self.logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                    return []
                
                # JSON 응답 파싱
                data = response.json()
                
                # 검색 결과가 없는 경우
                if data.get('total', 0) == 0 or not data.get('items'):
                    self.logger.info(f"No results found for '{query}' on page {page}")
                    return []
                
                # 결과 요약 로깅
                self.logger.info(f"총 검색 결과: {data.get('total', 0)}개, 현재 페이지 아이템: {len(data.get('items', []))}개")
                
                # 제품 데이터 변환
                products = []
                for item in data.get('items', []):
                    product = await self._convert_api_item_to_product(item, query)
                    if product:
                        products.append(product)
                
                # 캐시에 저장
                if products:
                    self.add_sparse_data(cache_key, products, ttl=3600)  # 1시간 캐싱
                
                return products
                
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(f"Error during API request: {str(e)}. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to fetch API results after {max_retries} retries: {str(e)}")
                    return []
        
        return []
    
    async def _convert_api_item_to_product(self, item: Dict, query: str) -> Optional[Product]:
        """API 응답 아이템을 Product 객체로 변환"""
        try:
            # HTML 태그 제거
            title = re.sub(r'<[^>]+>', '', item.get('title', ''))
            
            # 가격 추출
            price = int(item.get('lprice', 0))
            
            # 제품 ID 생성
            product_id = item.get('productId', '') or hashlib.md5(item.get('link', '').encode()).hexdigest()
            
            # 카테고리 정보
            category = item.get('category1', '')
            if item.get('category2'):
                category += f" > {item.get('category2')}"
            if item.get('category3'):
                category += f" > {item.get('category3')}"
            if item.get('category4'):
                category += f" > {item.get('category4')}"
            
            # 판매처 정보
            mall_name = item.get('mallName', '')
            
            # 이미지 URL
            image_url = item.get('image', '')
            
            # 홍보성 제품 여부 확인
            is_promotional = self._is_promotional_product(title, mall_name, category)
            
            # Product 객체 생성
            product = Product(
                id=product_id,
                name=title,
                price=price,
                source='naver_api',
                url=item.get('link', ''),
                image_url=image_url,
                brand=mall_name,
                category=category,
                is_promotional_site=is_promotional,
                original_input_data=item
            )
            
            return product
        except Exception as e:
            self.logger.error(f"Error converting API item to Product: {str(e)}", exc_info=True)
            return None
    
    def _is_promotional_product(self, title: str, mall_name: str, category: str) -> bool:
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

class NaverShoppingCrawler(BaseMultiLayerScraper):
    """
    네이버 쇼핑 크롤러 - 공식 API와 연결하는 브릿지 클래스
    
    특징:
    - .env 파일 또는 설정에서 API 키 로드
    - NaverShoppingAPI 클래스의 래퍼
    - 재시도 및 예외 처리
    """
    
    def __init__(self, max_retries: int = 3, cache: Optional[FileCache] = None, 
                 timeout: int = 30, use_proxies: bool = False):
        super().__init__(max_retries=max_retries, cache=cache, timeout=timeout)
        
        # API 키 로드
        self.api_keys = self._load_api_keys()
        
        # Naver Shopping API 인스턴스 생성
        self.api = NaverShoppingAPI(
            client_id=self.api_keys['client_id'],
            client_secret=self.api_keys['client_secret'],
            max_retries=max_retries,
            cache=cache,
            timeout=timeout
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # 프록시 설정 (필요 시)
        self.use_proxies = use_proxies
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        .env 파일이나 환경 변수에서 네이버 API 키 로드
        config.ini 파일에서도 대체 값을 찾음
        """
        import os
        from dotenv import load_dotenv
        
        # .env 파일 로드
        load_dotenv()
        
        # API 키 로드 시도
        client_id = os.getenv("client_id")
        client_secret = os.getenv("client_secret")
        
        # .env에서 찾지 못하면 config.ini 파일에서 찾기 시도
        if not client_id or not client_secret:
            try:
                config = configparser.ConfigParser()
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.ini')
                
                if os.path.exists(config_path):
                    config.read(config_path, encoding='utf-8')
                    
                    if 'API' in config and 'NAVER_CLIENT_ID' in config['API'] and 'NAVER_CLIENT_SECRET' in config['API']:
                        client_id = config['API']['NAVER_CLIENT_ID']
                        client_secret = config['API']['NAVER_CLIENT_SECRET']
                        self.logger.info("API 키를 config.ini 파일에서 로드했습니다.")
            except Exception as e:
                self.logger.warning(f"config.ini 파일에서 API 키를 로드하는 중 오류 발생: {e}")
        
        # 키가 없는 경우 로그에 오류 기록
        if not client_id or not client_secret:
            self.logger.error("네이버 API 키를 찾을 수 없습니다. .env 파일이나 config.ini에 client_id와 client_secret이 설정되어 있는지 확인하세요.")
            raise ValueError("네이버 API 키를 찾을 수 없습니다.")
        
        # 인증 성공 시 간단한 로깅
        self.logger.info(f"네이버 API 키 로드 성공 (client_id: {client_id[:4]}...)")
        
        return {
            "client_id": client_id,
            "client_secret": client_secret
        }
    
    def search_product(self, query: str, max_items: int = 50, reference_price: float = 0) -> List[Product]:
        """
        네이버 쇼핑에서 제품 검색 - API를 통해 검색 수행
        
        Args:
            query: 검색어
            max_items: 최대 검색 결과 수
            reference_price: 참조 가격 (10% 룰 적용용)
        
        Returns:
            List[Product]: 검색된 제품 목록
        """
        try:
            self.logger.info(f"네이버 쇼핑 검색 시작: '{query}'")

            # 가이드라인 반영: 상품명에서 '_'를 공백으로 치환
            processed_query = query.replace('_', ' ')
            if processed_query != query:
                self.logger.info(f"검색어 전처리: '{query}' -> '{processed_query}'")

            # NaverShoppingAPI의 search_product 메서드 호출 (처리된 검색어 사용)
            products = self.api.search_product(
                query=processed_query,
                max_items=max_items,
                reference_price=reference_price
            )

            if not products:
                self.logger.warning(f"'{processed_query}'에 대한 검색 결과가 없습니다.")
            else:
                self.logger.info(f"'{processed_query}'에 대한 검색 결과 {len(products)}개 발견")

            return products

        except Exception as e:
            self.logger.error(f"네이버 쇼핑 검색 중 오류 발생: {str(e)}", exc_info=True)
            # 빈 결과 반환
            return []
            
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
