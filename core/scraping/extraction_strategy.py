"""
데이터 추출 전략 클래스 모듈
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup

class ExtractionStrategy(ABC):
    """
    데이터 추출 전략의 기본 클래스
    """
    
    @abstractmethod
    def extract(self, source: Any) -> Optional[Dict[str, Any]]:
        """데이터를 추출합니다."""
        pass

class DOMExtractionStrategy(ExtractionStrategy):
    """
    DOM 기반 데이터 추출 전략
    """
    
    def extract(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """BeautifulSoup 객체에서 데이터를 추출합니다."""
        raise NotImplementedError

class TextExtractionStrategy(ExtractionStrategy):
    """
    텍스트 기반 데이터 추출 전략
    """
    
    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트에서 데이터를 추출합니다."""
        raise NotImplementedError

# Available extraction strategies
strategies = {
    "dom": DOMExtractionStrategy,
    "text": TextExtractionStrategy
} 