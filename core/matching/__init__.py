"""
Shop RPA Matching System

이 모듈은 상품 매칭을 위한 다양한 매처(Matcher)들을 제공합니다.
"""

__version__ = "1.0.0"
__author__ = "Shop RPA Team"
__last_updated__ = "2024-04-12"

from .image_matcher import ImageMatcher
from .text_matcher import TextMatcher
from .multimodal_matcher import MultiModalMatcher

__all__ = [
    "ImageMatcher",
    "TextMatcher",
    "MultiModalMatcher",
]
