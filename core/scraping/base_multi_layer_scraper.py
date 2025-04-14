"""
멀티 레이어 스크래퍼 클래스 모듈
"""

from typing import Any, Dict, List, Optional
import time
from .base_scraper import BaseScraper

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
        raise NotImplementedError
        
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
        raise NotImplementedError 