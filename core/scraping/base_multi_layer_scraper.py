"""
멀티 레이어 스크래퍼 클래스 모듈
"""

from typing import Any, Dict, List, Optional

from .base_scraper import BaseScraper

class BaseMultiLayerScraper(BaseScraper):
    """
    여러 계층의 데이터를 스크래핑하는 스크래퍼의 기본 클래스
    """
    
    def __init__(self, max_retries: int = 5, timeout: int = 30, cache: Optional[Any] = None):
        super().__init__(max_retries=max_retries, timeout=timeout, cache=cache)
        
    def get_sparse_data(self, key: str) -> Optional[Any]:
        """캐시에서 데이터를 가져옵니다."""
        if self.cache:
            return self.cache.get(key)
        return None
        
    def cache_sparse_data(self, key: str, data: Any, ttl: int = 86400) -> None:
        """데이터를 캐시에 저장합니다."""
        if self.cache:
            self.cache.set(key, data, ttl)
            
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """상품 정보를 가져옵니다."""
        raise NotImplementedError
        
    def search_product(self, query: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """상품을 검색합니다."""
        raise NotImplementedError 