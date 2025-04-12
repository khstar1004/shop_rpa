"""
베이스 스크래퍼 클래스 모듈
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseScraper(ABC):
    """
    모든 스크래퍼의 기본 클래스
    """
    
    def __init__(self, max_retries: int = 5, timeout: int = 30, cache: Optional[Any] = None):
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache = cache
        
    @abstractmethod
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """상품 정보를 가져옵니다."""
        pass
        
    @abstractmethod
    def search_product(self, query: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """상품을 검색합니다."""
        pass 