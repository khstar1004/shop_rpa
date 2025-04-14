"""
베이스 스크래퍼 클래스 모듈
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class BaseScraper(ABC):
    """
    모든 스크래퍼의 기본 클래스
    """
    
    def __init__(
        self, 
        max_retries: int = 3,  # 재시도 횟수 감소
        timeout: int = 15,     # 타임아웃 감소
        cache: Optional[Any] = None,
        connect_timeout: int = 5,
        read_timeout: int = 10
    ):
        self.max_retries = max_retries
        self.timeout = (connect_timeout, read_timeout)
        self.cache = cache
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 공유 세션 초기화
        self.session = self._create_optimized_session()
        
    def _create_optimized_session(self) -> requests.Session:
        """최적화된 요청 세션을 생성합니다."""
        session = requests.Session()
        
        # 재시도 전략 구성
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.3,  # 재시도 간격을 더 짧게 설정
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # 어댑터 구성
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
            pool_block=False
        )
        
        # 어댑터 마운트
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 기본 헤더 설정
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
        return session
        
    @abstractmethod
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """상품 정보를 가져옵니다."""
        pass
        
    @abstractmethod
    def search_product(self, query: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """상품을 검색합니다."""
        pass

    def fetch_url(
        self, 
        url: str, 
        params: Optional[Dict[str, Any]] = None, 
        use_session: bool = True,  # 기본값을 True로 변경
        timeout: Optional[Any] = None
    ) -> Optional[str]:
        """
        URL에서 콘텐츠를 가져옵니다.
        
        Args:
            url: 가져올 URL
            params: 요청 파라미터
            use_session: 세션 사용 여부 (기본값: True)
            timeout: 타임아웃 값 (기본값: None - 클래스 타임아웃 사용)
            
        Returns:
            str: 응답 텍스트 또는 실패 시 None
        """
        actual_timeout = timeout if timeout is not None else self.timeout
        
        try:
            if use_session:
                response = self.session.get(url, params=params, timeout=actual_timeout)
            else:
                # 새 요청 생성
                response = requests.get(url, params=params, timeout=actual_timeout)
                
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.RequestException as e:
            # 오류 로깅
            self.logger.warning(f"URL 가져오기 실패: {url}, 오류: {e}")
            return None 