import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
import re

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
        self.progress_callback = None  # 진행상황 콜백 초기화
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
                    if self.progress_callback:
                        self.progress_callback(progress_percent)
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
                try:
                    match_result = self._calculate_match_similarities(product, match)
                    processing_result.naver_matches.append(match_result)
                except Exception as match_err:
                    self.logger.warning(f"Error calculating similarities for {match.name}: {str(match_err)}")
                    continue
            
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
        
        # 데이터 검증 및 비어있는 필드 처리
        self._ensure_valid_result(processing_result)
        
        return processing_result
    
    def _ensure_valid_result(self, result: ProcessingResult) -> None:
        """결과 데이터가 유효한지 확인하고 필요한 기본값을 설정"""
        # 소스 제품 데이터 검증
        if not hasattr(result.source_product, 'original_input_data') or result.source_product.original_input_data is None:
            result.source_product.original_input_data = {}
        
        # 필수 필드 존재 확인
        required_fields = ['구분', '담당자', '업체명', '업체코드', '상품Code', '중분류카테고리', '상품명', 
                          '기본수량(1)', '판매단가(V포함)', '본사상품링크']
        
        for field in required_fields:
            if field not in result.source_product.original_input_data:
                result.source_product.original_input_data[field] = ''
            
        # 중요 필드 유효성 검사
        if 'Code' in result.source_product.original_input_data and not result.source_product.original_input_data.get('상품Code'):
            result.source_product.original_input_data['상품Code'] = result.source_product.original_input_data['Code']
        
        if '업체코드' in result.source_product.original_input_data and not result.source_product.original_input_data['업체코드']:
            if hasattr(result.source_product, 'product_code') and result.source_product.product_code:
                result.source_product.original_input_data['업체코드'] = result.source_product.product_code
    
    def _safe_naver_search(self, query: str) -> List[Product]:
        """안전하게 네이버 검색 실행"""
        products = []
        original_query = query
        
        # 검색 변형 시도들 (원래 검색어, 단어 수 줄이기)
        queries_to_try = []
        
        # 원래 검색어 추가
        queries_to_try.append(original_query)
        
        # 검색어에서 특수문자와 불필요한 정보 제거 (예: 규격, 수량 정보)
        cleaned_query = re.sub(r'[^\w\s]', ' ', original_query)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        if cleaned_query != original_query:
            queries_to_try.append(cleaned_query)
        
        # 단어 수 줄이기 (긴 검색어의 경우 앞쪽 3-4 단어만 사용)
        words = cleaned_query.split()
        if len(words) > 4:
            shortened_query = ' '.join(words[:4])
            queries_to_try.append(shortened_query)
        elif len(words) > 3:
            shortened_query = ' '.join(words[:3])
            queries_to_try.append(shortened_query)
        
        # 각 검색어 변형으로 시도
        max_attempts = 3  # 최대 변형 횟수 제한
        
        for attempt, current_query in enumerate(queries_to_try[:max_attempts]):
            if attempt > 0:
                self.logger.info(f"🔍 검색어 변형 시도 {attempt}: '{current_query}'")
            
            try:
                # 기본 호출 시도
                current_products = self.naver_crawler.search_product(current_query)
                
                if current_products:
                    products.extend(current_products)
                    self.logger.info(f"✅ 검색 성공: '{current_query}'에서 {len(current_products)}개 상품 발견")
                    
                    # 충분한 결과를 찾았으면 더 이상 시도하지 않음
                    if len(products) >= 10:
                        break
                else:
                    self.logger.warning(f"⚠️ '{current_query}'에 대한 검색 결과 없음")
            
            except TypeError as e:
                # 인자 오류 시 대체 호출
                self.logger.warning(f"Type error during Naver search: {str(e)}. Trying alternative method.")
                try:
                    # 직접 내부 검색 로직 호출
                    if hasattr(self.naver_crawler, '_search_product_logic'):
                        current_products = self.naver_crawler._search_product_logic(current_query, 50, None)
                        if current_products:
                            products.extend(current_products)
                            self.logger.info(f"✅ 대체 검색 성공: '{current_query}'에서 {len(current_products)}개 상품 발견")
                    elif hasattr(self.naver_crawler, '_search_product_async'):
                        # 비동기 호출 처리
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            current_products = loop.run_until_complete(
                                self.naver_crawler._search_product_async(current_query, 50, None)
                            )
                            if current_products:
                                products.extend(current_products)
                                self.logger.info(f"✅ 비동기 검색 성공: '{current_query}'에서 {len(current_products)}개 상품 발견")
                        finally:
                            loop.close()
                except Exception as inner_e:
                    self.logger.error(f"Alternative Naver search failed: {str(inner_e)}", exc_info=True)
            
            except Exception as e:
                self.logger.error(f"Error in Naver search: {str(e)}", exc_info=True)
        
        # 중복 제거
        unique_products = {}
        for product in products:
            if product.id not in unique_products:
                unique_products[product.id] = product
        
        # 결과가 없거나 충분하지 않은 경우 다른 검색 엔진이나 방법을 시도할 수 있음
        if not unique_products:
            self.logger.warning(f"❌ 모든 검색 시도 후에도 '{original_query}'에 대한 검색 결과 없음")
        else:
            self.logger.info(f"🎯 '{original_query}'에 대해 총 {len(unique_products)}개의 고유 상품 발견")
            
        return list(unique_products.values())
    
    def _calculate_match_similarities(self, source_product: Product, matched_product: Product) -> MatchResult:
        """두 제품 간의 유사도 계산"""
        # 텍스트 유사도
        text_sim = self.text_matcher.calculate_similarity(
            source_product.name,
            matched_product.name
        )
        
        # 이미지 유사도 계산 개선
        # 소스 이미지 URL 확인
        source_image_url = source_product.original_input_data.get('본사 이미지', '')
        if not source_image_url and source_product.image_url:
            source_image_url = source_product.image_url
            
        # 매치된 이미지 URL 확인
        match_image_url = matched_product.image_url or ''
        
        # 이미지 유사도 계산
        image_sim = 0.0  # 기본값
        if source_image_url and match_image_url:
            self.logger.debug(f"Calculating image similarity between: {source_image_url} and {match_image_url}")
            image_sim = self.image_matcher.calculate_similarity(source_image_url, match_image_url)
        else:
            # 소스와 매치 제공자를 명확히 표시하여 로깅
            if not source_image_url and match_image_url:
                # 원본 제품의 이미지가 없지만 매칭된 제품(네이버/고려)은 이미지가 있는 경우
                source_name = source_product.name
                match_source = "네이버" if matched_product.source == 'naver_api' else "고려기프트"
                
                # 이모지를 사용하여 구분하기 쉽게 만듦
                self.logger.warning(f"⚠️ 원본 제품 '{source_name}'에 이미지가 없지만, {match_source}에서 매칭된 제품에는 이미지가 있습니다: {match_image_url}")
                
                # 원본 제품에 이미지가 없는 경우에도 품질 분석에 크게 영향을 주지 않도록 이미지 유사도에 기본값 부여
                # 0.0은 너무 낮아 전체 매칭 점수를 낮출 수 있으므로 중간 점수인 0.5 부여
                if matched_product.source == 'naver_api':
                    # 네이버 API 결과는 항상 이미지를 포함하므로 더 높은 기본값 사용
                    image_sim = 0.5
            else:
                # 기타 이미지 누락 케이스 (둘 다 없거나 매칭된 제품 이미지가 없는 경우)
                self.logger.warning(f"Missing image URL for similarity calculation: source={bool(source_image_url)}, match={bool(match_image_url)}")
        
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
        """
        매칭 결과 중 최적의 결과 선택
        
        매뉴얼 요구사항:
        1. 상품 이름으로 검색하여 동일 상품 찾기
        2. 이미지로 제품 비교 (이미지 비교가 어려운 경우 규격 확인)
        3. 동일 상품으로 판단되면 가장 낮은 가격의 상품 선택
        """
        if not matches:
            return None
        
        # 임계값 설정
        text_threshold = self.config['MATCHING'].get('TEXT_SIMILARITY_THRESHOLD', 0.65)
        image_threshold = self.config['MATCHING'].get('IMAGE_SIMILARITY_THRESHOLD', 0.3)
        
        # 엄격한 매칭: 임계값 이상인 매칭만 필터링 (동일 상품으로 간주)
        valid_matches = [
            m for m in matches
            if m.text_similarity >= text_threshold
            and (
                # 이미지가 있는 경우는 이미지 유사도 검사
                (m.image_similarity >= image_threshold) or
                # 이미지가 없지만 텍스트 유사도가 매우 높은 경우(0.85 이상) 예외 허용
                (m.image_similarity == 0 and m.text_similarity >= 0.85)
            )
        ]
        
        # 임계값을 통과한 매칭이 있으면 그 중에서 최저가 선택
        if valid_matches:
            best_match = min(valid_matches, key=lambda x: x.matched_product.price if x.matched_product.price > 0 else float('inf'))
            self.logger.info(f"💯 엄격한 임계값을 통과한 최적의 매칭: {best_match.matched_product.name} (텍스트 유사도: {best_match.text_similarity:.2f}, 이미지 유사도: {best_match.image_similarity:.2f}, 가격: {best_match.matched_product.price})")
            return best_match
        
        # 임계값을 통과한 매칭이 없으면 더 낮은 임계값 시도
        relaxed_text_threshold = text_threshold * 0.75  # 25% 낮은 임계값
        relaxed_matches = [
            m for m in matches
            if m.text_similarity >= relaxed_text_threshold
        ]
        
        if relaxed_matches:
            # 낮은 임계값에서는 텍스트 유사도와 가격을 동시에 고려해 가장 적합한 매칭 선택
            # 텍스트 유사도로 정렬 후 상위 3개 중에서 최저가 선택
            top_matches = sorted(relaxed_matches, key=lambda x: x.text_similarity, reverse=True)[:3]
            best_match = min(top_matches, key=lambda x: x.matched_product.price if x.matched_product.price > 0 else float('inf'))
            
            self.logger.warning(f"⚠️ 낮은 임계값으로 매칭 발견: {best_match.matched_product.name} (텍스트 유사도: {best_match.text_similarity:.2f}, 이미지 유사도: {best_match.image_similarity:.2f}, 가격: {best_match.matched_product.price})")
            return best_match
        
        # 모든 임계값에서 매칭을 찾지 못한 경우, 모든 매칭 중 가장 유사한 하나 반환 (매우 유연한 대안)
        if matches:
            # 통합 유사도로 정렬해 가장 높은 하나 선택
            best_match = max(matches, key=lambda x: x.combined_similarity)
            self.logger.warning(f"❗ 모든 임계값 실패, 가장 유사한 제품 선택: {best_match.matched_product.name} (통합 유사도: {best_match.combined_similarity:.2f})")
            return best_match
        
        return None

    def process_files(self, input_files: List[str], output_dir: str = None, limit: int = None) -> List[str]:
        """여러 파일을 처리합니다."""
        try:
            if not input_files:
                self.logger.error("입력 파일이 없습니다.")
                return []
            
            # 출력 디렉토리 설정
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            output_files = []
            for input_file in input_files:
                try:
                    # 제한된 수의 상품만 처리
                    if limit:
                        output_file = self._process_limited_file(input_file, output_dir, limit)
                    else:
                        output_file = self._process_single_file(input_file, output_dir)
                    
                    if output_file:
                        output_files.append(output_file)
                    
                except Exception as e:
                    self.logger.error(f"파일 처리 중 오류 발생: {input_file}, 오류: {str(e)}", exc_info=True)
                    continue
                
            return output_files
        
        except Exception as e:
            self.logger.error(f"파일 처리 중 오류 발생: {str(e)}", exc_info=True)
            return []

    def _process_limited_file(self, input_file: str, output_dir: str = None, limit: int = 10) -> Optional[str]:
        """제한된 수의 상품만 처리합니다."""
        try:
            self.logger.info(f"파일 처리 시작: {input_file} (최대 {limit}개 상품)")
            
            # Excel 파일 읽기
            df = pd.read_excel(input_file)
            if df.empty:
                self.logger.error(f"파일이 비어있습니다: {input_file}")
                return None
            
            # 데이터 정제
            df = self._clean_data(df)
            
            # 제한된 수의 상품만 선택
            if len(df) > limit:
                self.logger.info(f"전체 {len(df)}개 중 {limit}개 상품만 처리합니다.")
                df = df.head(limit)
            
            # 결과 저장을 위한 리스트
            results = []
            
            # 각 상품 처리
            for idx, row in df.iterrows():
                try:
                    # 상품 처리
                    result = self._process_single_product(row)
                    if result:
                        results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"상품 처리 중 오류 발생 (행 {idx+1}): {str(e)}", exc_info=True)
                    continue
            
            # 결과가 있는 경우에만 출력 파일 생성
            if results:
                # 출력 파일명 생성
                if output_dir:
                    base_name = os.path.basename(input_file)
                    output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}-result.xlsx")
                else:
                    output_file = f"{os.path.splitext(input_file)[0]}-result.xlsx"
                
                # Excel 파일 생성
                self.excel_manager.generate_enhanced_output(results, output_file)
                
                self.logger.info(f"파일 처리 완료: {output_file}")
                return output_file
            
            return None
        
        except Exception as e:
            self.logger.error(f"파일 처리 중 오류 발생: {str(e)}", exc_info=True)
            return None 