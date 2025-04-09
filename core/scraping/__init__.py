"""
고급 웹 스크래핑 시스템

다중 레이어 추출 엔진과 성능 최적화 기능을 구현한 기본 스크래퍼 클래스들을 제공합니다.
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import re
from urllib.parse import urlparse
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from utils.caching import FileCache

class ExtractionStrategy(ABC):
    """추출 전략 인터페이스"""
    
    @abstractmethod
    def extract(self, source, selector, **kwargs):
        """주어진 소스에서 데이터 추출"""
        pass
    
    @abstractmethod
    def can_handle(self, source, selector, **kwargs) -> bool:
        """이 전략이 주어진 소스를 처리할 수 있는지 확인"""
        pass

class DOMExtractionStrategy(ExtractionStrategy):
    """DOM 기반 추출 전략"""
    
    def can_handle(self, source, selector, **kwargs) -> bool:
        return hasattr(source, 'select') or hasattr(source, 'find')
    
    def extract(self, source, selector, **kwargs):
        try:
            # BeautifulSoup 인터페이스
            if hasattr(source, 'select'):
                return source.select(selector)
            # Selenium 인터페이스
            elif hasattr(source, 'find_element'):
                if kwargs.get('multiple', False):
                    return source.find_elements(kwargs.get('by', 'css selector'), selector)
                else:
                    return source.find_element(kwargs.get('by', 'css selector'), selector)
            return None
        except Exception as e:
            logging.debug(f"DOM extraction failed: {str(e)}")
            return None

class TextExtractionStrategy(ExtractionStrategy):
    """텍스트 범위 기반 추출 전략"""
    
    def can_handle(self, source, selector, **kwargs) -> bool:
        return isinstance(source, str) and isinstance(selector, (str, re.Pattern))
    
    def extract(self, source, selector, **kwargs):
        try:
            if isinstance(selector, re.Pattern):
                matches = selector.findall(source)
                return matches if kwargs.get('multiple', False) else (matches[0] if matches else None)
            else:
                # 시작 및 종료 문자열 기반 추출
                start_marker = selector
                end_marker = kwargs.get('end_marker', None)
                
                if end_marker:
                    start_pos = source.find(start_marker)
                    if start_pos == -1:
                        return None
                    
                    start_pos += len(start_marker)
                    end_pos = source.find(end_marker, start_pos)
                    
                    if end_pos == -1:
                        return None
                    
                    return source[start_pos:end_pos]
                else:
                    # 정확한 텍스트 검색
                    return source if start_marker in source else None
        except Exception as e:
            logging.debug(f"Text extraction failed: {str(e)}")
            return None

class CoordinateExtractionStrategy(ExtractionStrategy):
    """좌표 기반 추출 전략"""
    
    def can_handle(self, source, selector, **kwargs) -> bool:
        return hasattr(source, 'get_screenshot_as_base64') and isinstance(selector, (tuple, list))
    
    def extract(self, source, selector, **kwargs):
        try:
            # 좌표는 (x1, y1, x2, y2) 형식
            from PIL import Image
            import io
            import base64
            
            # Selenium webdriver에서 스크린샷 캡처
            if hasattr(source, 'get_screenshot_as_base64'):
                screenshot = source.get_screenshot_as_base64()
                image_data = base64.b64decode(screenshot)
                image = Image.open(io.BytesIO(image_data))
                
                # 좌표로 이미지 잘라내기
                x1, y1, x2, y2 = selector
                cropped = image.crop((x1, y1, x2, y2))
                
                # 옵션에 따라 OCR 사용
                if kwargs.get('use_ocr', False):
                    try:
                        # pytesseract is an optional dependency
                        import pytesseract
                        return pytesseract.image_to_string(cropped, lang=kwargs.get('lang', 'kor+eng'))
                    except ImportError:
                        logging.warning("Tesseract OCR not available. Returning image only.")
                        return cropped
                else:
                    return cropped
            return None
        except Exception as e:
            logging.debug(f"Coordinate extraction failed: {str(e)}")
            return None

class BaseMultiLayerScraper:
    """다중 레이어 추출 엔진을 구현한 기본 스크래퍼 클래스"""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30, cache: Optional[FileCache] = None):
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache = cache
        self.sparse_data = {}
        self.sparse_data_ttl = {}
        
        # 추출 전략 등록
        self.extraction_strategies = [
            DOMExtractionStrategy(),
            TextExtractionStrategy(),
            CoordinateExtractionStrategy()
        ]
        
        # 성능 최적화를 위한 작업 실행기
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="Scraper")
        
        # 현재 활성 작업 추적
        self.active_tasks = set()
    
    def extract(self, source, selector, **kwargs):
        """다중 레이어 추출 시스템으로 데이터 추출"""
        # 여러 전략을 시도
        for strategy in self.extraction_strategies:
            if strategy.can_handle(source, selector, **kwargs):
                result = strategy.extract(source, selector, **kwargs)
                if result is not None:
                    return result
        
        self.logger.debug(f"All extraction strategies failed for selector: {selector}")
        return None
    
    def _get_extraction_strategy(self, source, selector, **kwargs):
        """최적의 추출 전략 결정"""
        for strategy in self.extraction_strategies:
            if strategy.can_handle(source, selector, **kwargs):
                return strategy
        return None
    
    @lru_cache(maxsize=100)
    def get_cached_result(self, url, selector_key):
        """캐시된 결과 가져오기 (내부 메모리 캐시 사용)"""
        if self.cache:
            cache_key = f"scraper|{url}|{selector_key}"
            return self.cache.get(cache_key)
        return None
    
    def set_cached_result(self, url, selector_key, value, ttl=None):
        """결과 캐싱 (내부 메모리 캐시 사용)"""
        if self.cache:
            cache_key = f"scraper|{url}|{selector_key}"
            try:
                # Pass ttl to the cache if it's provided
                if ttl:
                    self.cache.set(cache_key, value, ttl=ttl)
                else:
                    self.cache.set(cache_key, value)
            except Exception as e:
                logging.error(f"Failed to cache results for key '{cache_key}': {str(e)}")
    
    async def extract_async(self, source, selectors, **kwargs):
        """여러 셀렉터를 비동기로 추출"""
        tasks = []
        
        for selector_key, selector_info in selectors.items():
            selector = selector_info['selector']
            selector_kwargs = selector_info.get('options', {})
            selector_kwargs.update(kwargs)
            
            # 적절한 전략 결정
            strategy = self._get_extraction_strategy(source, selector, **selector_kwargs)
            if not strategy:
                continue
                
            # 비동기 작업 생성
            task = asyncio.create_task(
                self._extract_with_timeout(strategy, source, selector, selector_kwargs)
            )
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
            tasks.append((selector_key, task))
        
        # 결과 수집
        results = {}
        for selector_key, task in tasks:
            try:
                results[selector_key] = await task
            except asyncio.TimeoutError:
                self.logger.warning(f"Extraction timed out for selector: {selector_key}")
                results[selector_key] = None
            except Exception as e:
                self.logger.error(f"Error extracting {selector_key}: {str(e)}")
                results[selector_key] = None
        
        return results
    
    async def _extract_with_timeout(self, strategy, source, selector, kwargs):
        """타임아웃을 적용한 비동기 추출"""
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                self.executor,
                lambda: strategy.extract(source, selector, **kwargs)
            ),
            timeout=kwargs.get('timeout', self.timeout)
        )
    
    def add_sparse_data(self, key, value, ttl=None):
        """희소 데이터 구조에 데이터 추가"""
        expire_time = time.time() + ttl if ttl else None
        self.sparse_data[key] = {
            'value': value,
            'expire_at': expire_time
        }
        
        # Also store in external cache if available
        if self.cache:
            try:
                # Pass ttl to the cache if it's provided
                if ttl:
                    self.cache.set(key, value, ttl=ttl)
                else:
                    self.cache.set(key, value)
            except Exception as e:
                logging.error(f"Failed to cache results for query '{key}': {str(e)}")
    
    def get_sparse_data(self, key):
        """희소 데이터 구조에서 데이터 가져오기"""
        if key not in self.sparse_data:
            return None
        
        data_obj = self.sparse_data[key]
        # TTL 확인
        if data_obj.get('expire_at') and data_obj['expire_at'] < time.time():
            # 만료된 데이터 삭제
            del self.sparse_data[key]
            return None
        
        return data_obj['value']
    
    def clean_expired_data(self):
        """만료된 희소 데이터 정리"""
        current_time = time.time()
        keys_to_remove = [
            key for key, data in self.sparse_data.items()
            if data['expire_at'] and current_time > data['expire_at']
        ]
        
        for key in keys_to_remove:
            del self.sparse_data[key]
    
    def shutdown(self):
        """자원 정리"""
        self.executor.shutdown(wait=True)
        
        # 실행 중인 모든 작업 취소
        for task in self.active_tasks:
            task.cancel()

# 공개 클래스/함수들
from .koryo_scraper import KoryoScraper
from .naver_crawler import NaverShoppingCrawler
from .haeoeum_scraper import HaeoeumScraper

__all__ = [
    'BaseMultiLayerScraper',
    'KoryoScraper',
    'NaverShoppingCrawler',
    'HaeoeumScraper'
]
