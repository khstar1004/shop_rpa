"""
멀티 레이어 스크래퍼 클래스 모듈
"""

from typing import Any, Dict, List, Optional
import time
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup

class BaseMultiLayerScraper(BaseScraper):
    """
    여러 계층의 데이터를 스크래핑하는 스크래퍼의 기본 클래스
    """
    
    def __init__(
        self, 
        max_retries: int = 3,  # 재시도 횟수 감소
        timeout: int = 15,     # 타임아웃 감소
        cache: Optional[Any] = None,
        cache_ttl: int = 3600  # 캐시 TTL 1시간으로 감소
    ):
        super().__init__(max_retries=max_retries, timeout=timeout, cache=cache)
        self.cache_ttl = cache_ttl
        
    def get_sparse_data(self, key: str) -> Optional[Any]:
        """캐시에서 데이터를 가져옵니다."""
        if not self.cache:
            return None
            
        data = self.cache.get(key)
        if not data:
            return None
            
        # TTL 확인
        timestamp = data.get('timestamp', 0)
        if time.time() - timestamp > self.cache_ttl:
            # TTL 만료된 데이터 삭제
            self.cache.delete(key)
            return None
            
        return data.get('value')
        
    def cache_sparse_data(self, key: str, data: Any) -> None:
        """데이터를 캐시에 저장합니다."""
        if not self.cache:
            return
            
        self.cache.set(key, {
            'value': data,
            'timestamp': time.time()
        })
            
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """상품 정보를 가져옵니다."""
        try:
            # 캐시에서 먼저 확인
            cache_key = f"product_{self.__class__.__name__.lower()}_{product_id}"
            cached_data = self.get_sparse_data(cache_key)
            if cached_data:
                return cached_data
                
            # 실제 상품 정보 가져오기
            product_data = self._get_product_impl(product_id)
            
            if product_data:
                # 캐시에 저장
                self.cache_sparse_data(cache_key, product_data)
                
            return product_data
            
        except Exception as e:
            self.logger.error(f"Failed to get product {product_id}: {str(e)}")
            return None
            
    def search_product(
        self, 
        query: str, 
        max_items: int = 50,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """상품을 검색합니다."""
        if use_cache:
            cache_key = f"search_{self.__class__.__name__.lower()}_{query}"
            cached_result = self.get_sparse_data(cache_key)
            if cached_result:
                return cached_result
                
        result = self._search_product_impl(query, max_items)
        
        if use_cache and result:
            self.cache_sparse_data(cache_key, result)
            
        return result
        
    def _search_product_impl(
        self, 
        query: str, 
        max_items: int = 50
    ) -> List[Dict[str, Any]]:
        """실제 검색 구현. 하위 클래스에서 구현해야 함."""
        try:
            # 검색 결과를 가져오는 기본 구현
            search_url = self._get_search_url(query)
            response = self._make_request(search_url)
            
            if not response:
                return []
                
            # 검색 결과 파싱
            soup = BeautifulSoup(response.text, 'html.parser')
            products = []
            
            # 검색 결과에서 상품 정보 추출
            for item in self._extract_search_items(soup, max_items):
                product_id = self._extract_product_id(item)
                if product_id:
                    product = self.get_product(product_id)
                    if product:
                        products.append(product)
                        
                if len(products) >= max_items:
                    break
                    
            return products
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {str(e)}")
            return []
            
    def _get_search_url(self, query: str) -> str:
        """검색 URL을 생성합니다."""
        raise NotImplementedError
        
    def _extract_search_items(self, soup: BeautifulSoup, max_items: int) -> List[Any]:
        """검색 결과에서 상품 항목을 추출합니다."""
        raise NotImplementedError
        
    def _extract_product_id(self, item: Any) -> Optional[str]:
        """상품 항목에서 상품 ID를 추출합니다."""
        raise NotImplementedError
        
    def _get_product_impl(self, product_id: str) -> Optional[Dict[str, Any]]:
        """실제 상품 정보를 가져옵니다."""
        raise NotImplementedError 