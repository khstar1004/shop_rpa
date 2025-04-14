"""
데이터 추출 전략 클래스 모듈
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import re

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
    
    def __init__(self, selectors: Dict[str, str]):
        """
        Args:
            selectors: CSS 선택자 딕셔너리 (필드명: 선택자)
        """
        self.selectors = selectors
    
    def extract(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """BeautifulSoup 객체에서 데이터를 추출합니다."""
        try:
            result = {}
            for field, selector in self.selectors.items():
                element = soup.select_one(selector)
                if element:
                    result[field] = element.get_text(strip=True)
                else:
                    result[field] = None
            return result
        except Exception as e:
            logging.error(f"DOM extraction failed: {str(e)}")
            return None

class TextExtractionStrategy(ExtractionStrategy):
    """
    텍스트 기반 데이터 추출 전략
    """
    
    def __init__(self, patterns: Dict[str, str]):
        """
        Args:
            patterns: 정규식 패턴 딕셔너리 (필드명: 패턴)
        """
        self.patterns = patterns
    
    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트에서 데이터를 추출합니다."""
        try:
            result = {}
            for field, pattern in self.patterns.items():
                match = re.search(pattern, text)
                if match:
                    result[field] = match.group(1)
                else:
                    result[field] = None
            return result
        except Exception as e:
            logging.error(f"Text extraction failed: {str(e)}")
            return None

# Available extraction strategies
strategies = {
    "dom": DOMExtractionStrategy,
    "text": TextExtractionStrategy
} 