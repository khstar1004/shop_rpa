import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os

import pandas as pd

from ..data_models import Product, MatchResult, ProcessingResult
from ..matching.text_matcher import TextMatcher
from ..matching.image_matcher import ImageMatcher
from ..matching.multimodal_matcher import MultiModalMatcher
from ..scraping.koryo_scraper import KoryoScraper
from ..scraping.naver_crawler import NaverShoppingCrawler
from utils.caching import FileCache

from .excel_manager import ExcelManager
from .data_cleaner import DataCleaner
from .product_factory import ProductFactory
from .file_splitter import FileSplitter

class ProductProcessor:
    """제품 데이터 처리를 위한 메인 클래스"""
    
    def __init__(self, config: Dict):
        """
        제품 프로세서 초기화
        
        Args:
            config: 애플리케이션 설정
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 하위 컴포넌트 초기화
        self._init_components()
        
        # 배치 처리 설정
        self.batch_size = config['PROCESSING'].get('BATCH_SIZE', 10)
    
    def _init_components(self):
        """필요한 컴포넌트 초기화"""
        # 캐시 초기화
        self.cache = FileCache(
            cache_dir=self.config['PATHS']['CACHE_DIR'],
            duration_seconds=self.config['PROCESSING']['CACHE_DURATION'],
            max_size_mb=self.config['PROCESSING'].get('CACHE_MAX_SIZE_MB', 1024),
            enable_compression=self.config['PROCESSING'].get('ENABLE_COMPRESSION', False),
            compression_level=self.config['PROCESSING'].get('COMPRESSION_LEVEL', 6)
        )
        
        # 매칭 컴포넌트 초기화
        self.text_matcher = TextMatcher(cache=self.cache)
        
        self.image_matcher = ImageMatcher(
            cache=self.cache,
            similarity_threshold=self.config['MATCHING']['IMAGE_SIMILARITY_THRESHOLD']
        )
        
        self.multimodal_matcher = MultiModalMatcher(
            text_weight=self.config['MATCHING']['TEXT_WEIGHT'],
            image_weight=self.config['MATCHING']['IMAGE_WEIGHT'],
            text_matcher=self.text_matcher,
            image_matcher=self.image_matcher,
            similarity_threshold=self.config['MATCHING'].get('TEXT_SIMILARITY_THRESHOLD', 0.75)
        )
        
        # 스크래퍼 초기화
        self.koryo_scraper = KoryoScraper(
            max_retries=self.config['PROCESSING']['MAX_RETRIES'],
            cache=self.cache,
            timeout=self.config['PROCESSING'].get('REQUEST_TIMEOUT', 30)
        )
        
        # 프록시 사용 여부 확인
        use_proxies = False
        if 'NETWORK' in self.config and self.config['NETWORK'].get('USE_PROXIES') == 'True':
            use_proxies = True
            self.logger.info("프록시 사용 모드로 네이버 크롤러를 초기화합니다.")
            
        self.naver_crawler = NaverShoppingCrawler(
            max_retries=self.config['PROCESSING']['MAX_RETRIES'],
            cache=self.cache,
            timeout=self.config['PROCESSING'].get('REQUEST_TIMEOUT', 30),
            use_proxies=use_proxies
        )
        
        # 스크래퍼 설정 적용
        scraping_config = self.config.get('SCRAPING', {})
        if scraping_config:
            self._configure_scrapers(scraping_config)
        
        # 스레드풀 초기화
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['PROCESSING']['MAX_WORKERS'],
            thread_name_prefix='ProductProcessor'
        )
        
        # 유틸리티 컴포넌트 초기화
        self.excel_manager = ExcelManager(self.config, self.logger)
        self.data_cleaner = DataCleaner(self.config, self.logger)
        self.product_factory = ProductFactory(self.config, self.logger, self.data_cleaner)
        self.file_splitter = FileSplitter(self.config, self.logger)
    
    def _configure_scrapers(self, scraping_config: Dict):
        """스크래퍼 설정 적용"""
        scrapers = [self.koryo_scraper, self.naver_crawler]
        
        for scraper in scrapers:
            # Max workers 설정
            max_workers = scraping_config.get('MAX_CONCURRENT_REQUESTS', 5)
            if hasattr(scraper, 'executor') and hasattr(scraper.executor, '_max_workers'):
                scraper.executor._max_workers = max_workers
            
            # Timeout 설정
            if hasattr(scraper, 'timeout'):
                scraper.timeout = scraping_config.get('EXTRACTION_TIMEOUT', 15)
            
            # Extraction strategies 설정
            if hasattr(scraper, 'extraction_strategies'):
                strategies = []
                
                if scraping_config.get('ENABLE_DOM_EXTRACTION', True):
                    for strategy in scraper.extraction_strategies:
                        if 'DOMExtractionStrategy' in strategy.__class__.__name__:
                            strategies.append(strategy)
                            
                if scraping_config.get('ENABLE_TEXT_EXTRACTION', True):
                    for strategy in scraper.extraction_strategies:
                        if 'TextExtractionStrategy' in strategy.__class__.__name__:
                            strategies.append(strategy)
                            
                if scraping_config.get('ENABLE_COORD_EXTRACTION', True):
                    for strategy in scraper.extraction_strategies:
                        if 'CoordinateExtractionStrategy' in strategy.__class__.__name__:
                            strategies.append(strategy)
                
                if strategies:
                    scraper.extraction_strategies = strategies
            
            # Politeness delay 설정
            if hasattr(scraper, '_search_product_async'):
                # 기존 메서드 수정하지 않고 설정 적용
                original_method = scraper._search_product_async
                politeness_delay = scraping_config.get('POLITENESS_DELAY', 1500) / 1000  # ms → 초
                
                async def patched_method(query, max_items=50, reference_price=None):
                    # politeness_delay 확인 로깅
                    scraper.logger.debug(f"Using politeness delay of {politeness_delay} seconds")
                    # 원래 메서드 호출
                    result = await original_method(query, max_items, reference_price)
                    return result
                
                scraper._search_product_async = patched_method
    
    def process_file(self, input_file: str) -> Tuple[Optional[str], Optional[str]]:
        """
        입력 엑셀 파일 처리 및 보고서 생성
        
        Args:
            input_file: 입력 엑셀 파일 경로
            
        Returns:
            (결과 파일 경로, 오류 메시지)
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"Processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 파일 존재 확인
            if not os.path.exists(input_file):
                error_msg = f"Input file not found: {input_file}"
                self.logger.error(error_msg)
                return None, error_msg
            
            # 입력 파일 읽기
            try:
                df = self.excel_manager.read_excel_file(input_file)
                
                if df.empty:
                    error_msg = "No data found in input file"
                    self.logger.error(error_msg)
                    return None, error_msg
                    
                total_items = len(df)
                self.logger.info(f"Loaded {total_items} items from {input_file}")
                
                # 데이터 정제
                df = self.data_cleaner.clean_dataframe(df)
                
            except Exception as e:
                self.logger.error(f"Failed to read input file: {str(e)}", exc_info=True)
                return None, f"Failed to read input file: {str(e)}"
            
            # 대용량 파일 분할 처리
            if self.file_splitter.needs_splitting(df):
                try:
                    split_files = self.file_splitter.split_input_file(df, input_file)
                    self.logger.info(f"Input file split into {len(split_files)} files")
                    
                    # 각 분할 파일 처리
                    result_files = []
                    for split_file in split_files:
                        result_file, _ = self._process_single_file(split_file)
                        if result_file:
                            result_files.append(result_file)
                    
                    # 결과 병합
                    if len(result_files) > 1:
                        merged_result = self.file_splitter.merge_result_files(result_files, input_file)
                        return merged_result, None
                    
                    return result_files[0] if result_files else None, None
                    
                except Exception as e:
                    self.logger.error(f"Error splitting input file: {str(e)}", exc_info=True)
                    # 단일 파일로 처리
                    self.logger.info("Falling back to processing as a single file")
                    return self._process_single_file(input_file)
            else:
                # 단일 파일 처리
                return self._process_single_file(input_file)
                
        except Exception as e:
            self.logger.error(f"Error in process_file: {str(e)}", exc_info=True)
            return None, str(e)
    
    def _process_single_file(self, input_file: str) -> Tuple[Optional[str], Optional[str]]:
        """단일 입력 파일 처리"""
        try:
            start_time = datetime.now()
            
            # 엑셀 파일 읽기
            df = self.excel_manager.read_excel_file(input_file)
            total_items = len(df)
            
            # 데이터 정제
            df = self.data_cleaner.clean_dataframe(df)
            
            # 배치 단위로 처리
            results = []
            processed_count = 0
            
            for i in range(0, total_items, self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                batch_futures = []
                
                # 각 행에 대해 Product 생성 및 처리 시작
                for _, row in batch.iterrows():
                    product = self.product_factory.create_product_from_row(row)
                    if product:  # 유효한 제품만 처리
                        future = self.executor.submit(self._process_single_product, product)
                        batch_futures.append((product, future))
                
                # 배치 완료 대기
                for product, future in batch_futures:
                    try:
                        result = future.result(timeout=300)  # 5분 타임아웃
                        results.append(result)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing product {product.id}: {str(e)}", 
                            exc_info=True
                        )
                        # 실패한 결과도 추가 (순서 유지)
                        results.append(ProcessingResult(
                            source_product=product,
                            error=str(e)
                        ))
                    
                    # 진행 상황 업데이트
                    processed_count += 1
                    progress_percent = int((processed_count / total_items) * 100)
                    if processed_count % 10 == 0 or processed_count == total_items:
                        self.logger.info(f"Progress: {processed_count}/{total_items} ({progress_percent}%)")
            
            # 결과 보고서 생성
            output_file = self.excel_manager.generate_enhanced_output(results, input_file)
            
            # 처리 완료 로깅
            end_time = datetime.now()
            processing_time = end_time - start_time
            self.logger.info(f"Processing finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total processing time: {processing_time}")
            
            return output_file, None
            
        except Exception as e:
            self.logger.error(f"Error in _process_single_file: {str(e)}", exc_info=True)
            return None, str(e)
    
    def _process_single_product(self, product: Product) -> ProcessingResult:
        """단일 제품 처리"""
        return self.process_product(product)
    
    def process_product(self, product: Product) -> ProcessingResult:
        """
        단일 제품의 매칭 처리
        
        Args:
            product: 처리할 Product 객체
            
        Returns:
            ProductResult 객체
        """
        self.logger.info(f"Processing product: {product.name} (ID: {product.id})")
        processing_result = ProcessingResult(source_product=product)
        
        # 고려기프트 매칭 검색
        try:
            koryo_matches = self.koryo_scraper.search_product(product.name)
            self.logger.debug(f"Found {len(koryo_matches)} Koryo matches for {product.name}")
            
            # 매칭 결과 계산
            for match in koryo_matches:
                match_result = self._calculate_match_similarities(product, match)
                processing_result.koryo_matches.append(match_result)
            
            # 최적 매칭 찾기
            processing_result.best_koryo_match = self._find_best_match(processing_result.koryo_matches)
            
            if processing_result.best_koryo_match:
                self.logger.info(
                    f"Best Koryo match for {product.name}: "
                    f"{processing_result.best_koryo_match.matched_product.name} "
                    f"({processing_result.best_koryo_match.combined_similarity:.2f})"
                )
        except Exception as e:
            self.logger.error(f"Error finding Koryo matches for {product.name}: {str(e)}", exc_info=True)
            processing_result.error = f"Koryo search error: {str(e)}"
        
        # 네이버 매칭 검색
        try:
            naver_matches = self._safe_naver_search(product.name)
            self.logger.debug(f"Found {len(naver_matches)} Naver matches for {product.name}")
            
            # 매칭 결과 계산
            for match in naver_matches:
                match_result = self._calculate_match_similarities(product, match)
                processing_result.naver_matches.append(match_result)
            
            # 최적 매칭 찾기
            processing_result.best_naver_match = self._find_best_match(processing_result.naver_matches)
            
            if processing_result.best_naver_match:
                self.logger.info(
                    f"Best Naver match for {product.name}: "
                    f"{processing_result.best_naver_match.matched_product.name} "
                    f"({processing_result.best_naver_match.combined_similarity:.2f})"
                )
        except Exception as e:
            self.logger.error(f"Error finding Naver matches for {product.name}: {str(e)}", exc_info=True)
            if not processing_result.error:  # 이전 오류가 없을 경우만 설정
                processing_result.error = f"Naver search error: {str(e)}"
        
        return processing_result
    
    def _safe_naver_search(self, query: str) -> List[Product]:
        """안전하게 네이버 검색 실행"""
        try:
            # 기본 호출 시도
            return self.naver_crawler.search_product(query)
        except TypeError as e:
            # 인자 오류 시 대체 호출
            self.logger.warning(f"Type error during Naver search: {str(e)}. Trying alternative method.")
            try:
                # 직접 내부 검색 로직 호출
                if hasattr(self.naver_crawler, '_search_product_logic'):
                    return self.naver_crawler._search_product_logic(query, 50, None)
                elif hasattr(self.naver_crawler, '_search_product_async'):
                    # 비동기 호출 처리
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        matches = loop.run_until_complete(
                            self.naver_crawler._search_product_async(query, 50, None)
                        )
                        return matches
                    finally:
                        loop.close()
            except Exception as inner_e:
                self.logger.error(f"Alternative Naver search failed: {str(inner_e)}", exc_info=True)
            # 모든 방법 실패 시
            return []
        except Exception as e:
            self.logger.error(f"Error in Naver search: {str(e)}", exc_info=True)
            return []
    
    def _calculate_match_similarities(self, source_product: Product, matched_product: Product) -> MatchResult:
        """두 제품 간의 유사도 계산"""
        # 텍스트 유사도
        text_sim = self.text_matcher.calculate_similarity(
            source_product.name,
            matched_product.name
        )
        
        # 이미지 유사도
        image_sim = self.image_matcher.calculate_similarity(
            source_product.original_input_data.get('본사 이미지', ''),  # 원본 이미지 URL
            matched_product.image_url
        )
        
        # 통합 유사도
        combined_sim = self.multimodal_matcher.calculate_similarity(
            text_sim,
            image_sim
        )
        
        self.logger.debug(
            f"  Match candidate {matched_product.name} ({matched_product.source}): "
            f"Txt={text_sim:.2f}, Img={image_sim:.2f}, Comb={combined_sim:.2f}"
        )
        
        # 가격 차이 계산
        price_diff = 0.0
        price_diff_percent = 0.0
        source_price = source_product.price
        
        if source_price and source_price > 0 and isinstance(matched_product.price, (int, float)):
            price_diff = matched_product.price - source_price
            price_diff_percent = (price_diff / source_price) * 100 if source_price != 0 else 0
        
        return MatchResult(
            source_product=source_product,
            matched_product=matched_product,
            text_similarity=text_sim,
            image_similarity=image_sim,
            combined_similarity=combined_sim,
            price_difference=price_diff,
            price_difference_percent=price_diff_percent
        )
    
    def _find_best_match(self, matches: List[MatchResult]) -> Optional[MatchResult]:
        """매칭 결과 중 최적의 결과 선택"""
        if not matches:
            return None
        
        # 임계값 설정
        text_threshold = self.config['MATCHING']['TEXT_SIMILARITY_THRESHOLD']
        image_threshold = self.config['MATCHING']['IMAGE_SIMILARITY_THRESHOLD']
        
        # 임계값 이상인 매칭만 필터링
        valid_matches = [
            m for m in matches
            if m.text_similarity >= text_threshold
            and m.image_similarity >= image_threshold
        ]
        
        if not valid_matches:
            return None
        
        # 통합 유사도가 가장 높은 결과 반환
        return max(valid_matches, key=lambda x: x.combined_similarity) 