"""
스크래핑 모듈 초기화
"""

from .base_scraper import BaseScraper
from .base_multi_layer_scraper import BaseMultiLayerScraper
from .extraction_strategy import DOMExtractionStrategy, TextExtractionStrategy
from .utils import extract_main_image

# 스크래퍼 클래스 임포트
from .koryo_scraper import KoryoScraper
from .haeoeum_scraper import HaeoeumScraper
from .naver_crawler import NaverShoppingCrawler

__all__ = [
    "BaseScraper",
    "BaseMultiLayerScraper",
    "DOMExtractionStrategy",
    "TextExtractionStrategy",
    "extract_main_image",
    "KoryoScraper",
    "HaeoeumScraper",
    "NaverShoppingCrawler",
]
