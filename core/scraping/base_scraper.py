"""
베이스 스크래퍼 클래스 모듈
"""

import logging
import configparser
import os
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
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        cache: Optional[Any] = None,
        connect_timeout: Optional[int] = None,
        read_timeout: Optional[int] = None
    ):
        # Load configuration
        self.config = self._load_config()
        
        # Use provided values or fall back to config values
        self.max_retries = max_retries or int(self.config.get('SCRAPING', 'max_retries', fallback='3'))
        self.timeout = (connect_timeout or int(self.config.get('SCRAPING', 'connect_timeout', fallback='30')),
                       read_timeout or int(self.config.get('SCRAPING', 'read_timeout', fallback='30')))
        self.cache = cache
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', 'level', fallback='INFO')),
            format=self.config.get('logging', 'format', fallback='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # 공유 세션 초기화
        self.session = self._create_optimized_session()
        
    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini"""
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        config.read(config_path, encoding='utf-8')
        return config
        
    def _create_optimized_session(self) -> requests.Session:
        """최적화된 요청 세션을 생성합니다."""
        session = requests.Session()
        
        # 재시도 전략 구성
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=float(self.config.get('SCRAPING', 'backoff_factor', fallback='0.3')),
            status_forcelist=[int(x) for x in self.config.get('SCRAPING', 'retry_on_specific_status', fallback='429,503,502,500').split(',')],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # 어댑터 구성
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=int(self.config.get('SCRAPING', 'connection_pool_size', fallback='10')),
            pool_maxsize=int(self.config.get('SCRAPING', 'connection_pool_size', fallback='10')),
            pool_block=False
        )
        
        # 어댑터 마운트
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # SSL 검증 설정
        session.verify = self.config.getboolean('SCRAPING', 'ssl_verification', fallback=True)
        
        # 리다이렉트 설정
        session.max_redirects = int(self.config.get('SCRAPING', 'max_redirects', fallback='5'))
        
        # 기본 헤더 설정
        session.headers.update({
            "User-Agent": self.config.get('SCRAPING', 'user_agent', fallback='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
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