import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.caching import FileCache

from ..data_models import MatchResult, ProcessingResult, Product
from ..matching.image_matcher import ImageMatcher
from ..matching.multimodal_matcher import MultiModalMatcher
from ..matching.text_matcher import TextMatcher
from ..scraping.haeoeum_scraper import HaeoeumScraper
from ..scraping.koryo_scraper import KoryoScraper
from ..scraping.naver_crawler import NaverShoppingCrawler
from .data_cleaner import DataCleaner
from .excel_manager import ExcelManager
from .file_splitter import FileSplitter
from .product_factory import ProductFactory


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
        self._is_running = True  # Add running flag
        self._init_components()

        # 배치 처리 설정
        self.batch_size = config["PROCESSING"].get("BATCH_SIZE", 10)

    def stop_processing(self):
        """Stop the processing gracefully."""
        self.logger.info("Processing stop requested.")
        self._is_running = False

    def _init_components(self):
        """필요한 컴포넌트 초기화"""
        # 캐시 초기화
        self.cache = FileCache(
            cache_dir=self.config["PATHS"]["CACHE_DIR"],
            duration_seconds=self.config["PROCESSING"]["CACHE_DURATION"],
            max_size_mb=self.config["PROCESSING"].get("CACHE_MAX_SIZE_MB", 1024),
            enable_compression=self.config["PROCESSING"].get(
                "ENABLE_COMPRESSION", False
            ),
            compression_level=self.config["PROCESSING"].get("COMPRESSION_LEVEL", 6),
        )

        # 매칭 컴포넌트 초기화
        self.text_matcher = TextMatcher(cache=self.cache)

        # 이미지 처리 최대 해상도 설정 추가
        if "MAX_IMAGE_DIMENSION" not in self.config["MATCHING"]:
            self.config["MATCHING"]["MAX_IMAGE_DIMENSION"] = 256
            self.logger.info(f"Setting default MAX_IMAGE_DIMENSION to 256px")

        self.image_matcher = ImageMatcher(
            cache=self.cache,
            similarity_threshold=self.config["MATCHING"]["IMAGE_SIMILARITY_THRESHOLD"],
        )

        self.multimodal_matcher = MultiModalMatcher(
            text_weight=self.config["MATCHING"]["TEXT_WEIGHT"],
            image_weight=self.config["MATCHING"]["IMAGE_WEIGHT"],
            text_matcher=self.text_matcher,
            image_matcher=self.image_matcher,
            similarity_threshold=self.config["MATCHING"].get(
                "TEXT_SIMILARITY_THRESHOLD", 0.75
            ),
        )

        # 스크래퍼 초기화
        self.koryo_scraper = KoryoScraper(
            max_retries=self.config["PROCESSING"]["MAX_RETRIES"],
            cache=self.cache,
            timeout=self.config["PROCESSING"].get("REQUEST_TIMEOUT", 30),
        )

        # 프록시 사용 여부 확인
        use_proxies = False
        if (
            "NETWORK" in self.config
            and self.config["NETWORK"].get("USE_PROXIES") == "True"
        ):
            use_proxies = True
            self.logger.info("프록시 사용 모드로 네이버 크롤러를 초기화합니다.")

        self.naver_crawler = NaverShoppingCrawler(
            max_retries=self.config["PROCESSING"]["MAX_RETRIES"],
            cache=self.cache,
            timeout=self.config["PROCESSING"].get("REQUEST_TIMEOUT", 30),
        )

        # 스크래퍼 설정 적용
        scraping_config = self.config.get("SCRAPING", {})
        if scraping_config:
            self._configure_scrapers(scraping_config)

        # 최적화된 병렬 처리 설정
        # 시스템 코어 수 기반 max_workers 자동 설정
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        default_workers = max(4, min(cpu_count * 2, 16))  # 최소 4, 최대 16 워커

        # 스레드풀 초기화
        max_workers = self.config["PROCESSING"].get("MAX_WORKERS", default_workers)
        self.logger.info(f"병렬 처리 워커 수: {max_workers} (CPU 코어: {cpu_count})")
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ProductProcessor"
        )

        # 배치 크기 최적화
        default_batch = min(
            20, max(5, cpu_count)
        )  # 코어 수에 맞게 조정, 최소 5, 최대 20
        self.batch_size = self.config["PROCESSING"].get("BATCH_SIZE", default_batch)
        self.logger.info(f"배치 크기 설정: {self.batch_size}")

        # 유틸리티 컴포넌트 초기화
        self.excel_manager = ExcelManager(self.config, self.logger)
        self.data_cleaner = DataCleaner(self.config, self.logger)
        self.product_factory = ProductFactory(
            self.config, self.logger, self.data_cleaner
        )
        self.file_splitter = FileSplitter(self.config, self.logger)

    def _configure_scrapers(self, scraping_config: Dict):
        """스크래퍼 설정 적용"""
        scrapers = [self.koryo_scraper, self.naver_crawler]

        for scraper in scrapers:
            # Max workers 설정
            max_workers = scraping_config.get("MAX_CONCURRENT_REQUESTS", 5)
            if hasattr(scraper, "executor") and hasattr(
                scraper.executor, "_max_workers"
            ):
                scraper.executor._max_workers = max_workers

            # Timeout 설정
            if hasattr(scraper, "timeout"):
                scraper.timeout = scraping_config.get("EXTRACTION_TIMEOUT", 15)

            # Extraction strategies 설정
            if hasattr(scraper, "extraction_strategies"):
                strategies = []

                if scraping_config.get("ENABLE_DOM_EXTRACTION", True):
                    for strategy in scraper.extraction_strategies:
                        if "DOMExtractionStrategy" in strategy.__class__.__name__:
                            strategies.append(strategy)

                if scraping_config.get("ENABLE_TEXT_EXTRACTION", True):
                    for strategy in scraper.extraction_strategies:
                        if "TextExtractionStrategy" in strategy.__class__.__name__:
                            strategies.append(strategy)

                if scraping_config.get("ENABLE_COORD_EXTRACTION", True):
                    for strategy in scraper.extraction_strategies:
                        if (
                            "CoordinateExtractionStrategy"
                            in strategy.__class__.__name__
                        ):
                            strategies.append(strategy)

                if strategies:
                    scraper.extraction_strategies = strategies

            # Politeness delay 설정
            if hasattr(scraper, "_search_product_async"):
                # 기존 메서드 수정하지 않고 설정 적용
                original_method = scraper._search_product_async
                politeness_delay = (
                    scraping_config.get("POLITENESS_DELAY", 1500) / 1000
                )  # ms → 초

                async def patched_method(query, max_items=50, reference_price=None):
                    # politeness_delay 확인 로깅
                    scraper.logger.debug(
                        f"Using politeness delay of {politeness_delay} seconds"
                    )
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
            self.logger.info(
                f"Processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # 파일 존재 확인
            if not os.path.exists(input_file):
                error_msg = f"Input file not found: {input_file}"
                self.logger.error(error_msg)
                return None, error_msg

            # 엑셀 전처리 작업 (XLS -> XLSX 변환 및 필요한 컬럼 추가)
            input_file = self.process_excel_functionality(input_file)

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
                        merged_result = self.file_splitter.merge_result_files(
                            result_files, input_file
                        )
                        return merged_result, None

                    return result_files[0] if result_files else None, None

                except Exception as e:
                    self.logger.error(
                        f"Error splitting input file: {str(e)}", exc_info=True
                    )
                    # 단일 파일로 처리
                    self.logger.info("Falling back to processing as a single file")
                    return self._process_single_file(input_file)
            else:
                # 단일 파일 처리
                return self._process_single_file(input_file)

        except Exception as e:
            self.logger.error(f"Error in process_file: {str(e)}", exc_info=True)
            return None, str(e)

    def process_excel_functionality(self, input_file: str) -> str:
        """
        엑셀 파일에 대한 전처리 작업을 수행합니다.

        Args:
            input_file: 입력 엑셀 파일 경로

        Returns:
            str: 처리된 파일 경로 (변경이 있는 경우 새 파일 경로, 없으면 원본 경로)
        """
        try:
            input_dir = os.path.dirname(input_file)
            input_ext = os.path.splitext(input_file)[1].lower()

            # 1. XLS -> XLSX 변환 (확장자가 .xls인 경우)
            if input_ext == ".xls":
                self.logger.info(f"XLS 파일 감지: {input_file}")
                xlsx_file = self.excel_manager.convert_xls_to_xlsx(input_dir)
                if xlsx_file:
                    self.logger.info(f"XLS 파일이 XLSX로 변환되었습니다: {xlsx_file}")
                    input_file = xlsx_file
                else:
                    self.logger.warning(
                        "XLS 파일 변환에 실패했습니다. 원본 파일을 사용합니다."
                    )

            # 2. @ 기호 제거
            input_file = self.excel_manager.remove_at_symbol(input_file)

            # 3. 필요한 컬럼 확인 및 추가
            self.excel_manager.check_excel_file(input_file)

            return input_file

        except Exception as e:
            self.logger.error(f"엑셀 전처리 중 오류 발생: {str(e)}", exc_info=True)
            return input_file  # 오류 발생 시 원본 파일 사용

    def post_process_output_file(self, output_file: str) -> str:
        """
        출력 엑셀 파일에 대한 후처리 작업을 수행합니다.

        Args:
            output_file: 처리된 엑셀 파일 경로

        Returns:
            str: 최종 출력 파일 경로
        """
        try:
            # 1. 하이퍼링크 추가
            linked_file = self.excel_manager.add_hyperlinks_to_excel(output_file)

            # 2. 가격 차이가 있는 항목만 필터링
            filtered_file = self.excel_manager.filter_excel_by_price_diff(linked_file)

            # 3. 포맷팅 적용
            self.excel_manager.apply_formatting_to_excel(filtered_file)

            return filtered_file

        except Exception as e:
            self.logger.error(f"엑셀 후처리 중 오류 발생: {str(e)}", exc_info=True)
            return output_file

    def _process_single_file(
        self, input_file: str, output_dir: Optional[str] = None
    ) -> Optional[str]:
        """단일 입력 파일 처리"""
        # Reset running flag at the start of processing a file
        self._is_running = True
        try:
            start_time = datetime.now()
            self.logger.info(
                f"Processing file: {input_file}, started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # 엑셀 파일 읽기
            df = self.excel_manager.read_excel_file(input_file)
            if df.empty:
                self.logger.warning(f"Input file is empty: {input_file}")
                return None

            total_items = len(df)
            self.logger.info(f"Loaded {total_items} items from {input_file}")

            # 데이터 정제
            df = self.data_cleaner.clean_dataframe(df)

            # 배치 단위로 처리
            results = []
            processed_count = 0

            # Emit initial progress
            if self.progress_callback:
                try:
                    self.progress_callback(0, total_items)
                except Exception as cb_e:
                    self.logger.error(f"Error in initial progress callback: {cb_e}")

            for i in range(0, total_items, self.batch_size):
                # Check if stopped before starting a new batch
                if not self._is_running:
                    self.logger.warning("Processing stopped by request (batch loop).")
                    break

                batch = df.iloc[i : i + self.batch_size]
                batch_futures = []

                # 각 행에 대해 Product 생성 및 처리 시작
                for _, row in batch.iterrows():
                     # Check if stopped before submitting a new task
                    if not self._is_running:
                        self.logger.warning("Processing stopped by request (task submission loop).")
                        break # Break inner loop

                    product = self.product_factory.create_product_from_row(row)
                    if product:  # 유효한 제품만 처리
                        future = self.executor.submit(
                            self._process_single_product, product
                        )
                        batch_futures.append((product, future))
                
                # Check again if stopped after submitting tasks for the batch
                if not self._is_running:
                    break # Break outer loop if stopped during task submission

                # 배치 완료 대기
                for product, future in batch_futures:
                    # Check if stopped before getting result (allows faster stop)
                    if not self._is_running:
                        self.logger.warning("Processing stopped by request (result loop).")
                        # Attempt to cancel pending future if possible
                        if not future.done():
                            future.cancel()
                        continue # Skip getting result and updating progress for this item

                    try:
                        result = future.result(timeout=300)  # 5분 타임아웃
                        results.append(result)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing product {product.id}: {str(e)}",
                            exc_info=True,
                        )
                        # 실패한 결과도 추가 (순서 유지)
                        results.append(
                            ProcessingResult(source_product=product, error=str(e))
                        )
                    finally:
                        # Ensure progress is updated even if there was an error or stop request
                        processed_count += 1
                        if self.progress_callback:
                             try:
                                # Use processed_count and total_items
                                self.progress_callback(processed_count, total_items)
                             except Exception as cb_e:
                                 self.logger.error(f"Error in progress callback: {cb_e}")

                        if processed_count % 10 == 0 or processed_count == total_items:
                            progress_percent = int((processed_count / total_items) * 100) if total_items > 0 else 0
                            self.logger.info(
                                f"Progress: {processed_count}/{total_items} ({progress_percent}%)"
                            )
                
                # Check if stopped after processing the batch
                if not self._is_running:
                    break # Break outer loop if stopped

            # Check if processing was stopped before generating output
            if not self._is_running:
                self.logger.warning("Processing was stopped. Skipping output file generation.")
                return None

            # 결과 보고서 생성 (only if results exist and processing wasn't stopped)
            if results:
                output_file = self.excel_manager.generate_enhanced_output(
                    results, input_file, output_dir # Pass output_dir
                )

                # 후처리 작업 수행 (하이퍼링크, 필터링 등)
                output_file = self.post_process_output_file(output_file)

                # 처리 완료 로깅
                end_time = datetime.now()
                processing_time = end_time - start_time
                self.logger.info(
                    f"Processing finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.logger.info(f"Total processing time: {processing_time}")

                return output_file, None # Return tuple
            else:
                 self.logger.info("No results generated, possibly due to empty input or errors.")
                 return None, "No results generated" # Return tuple indicating no output

        except Exception as e:
            self.logger.error(f"Error in _process_single_file: {str(e)}", exc_info=True)
            return None, str(e) # Return tuple

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
        self.logger.info(f"Product source: {product.source}")

        # 해오름기프트 상품 확인
        if product.source.lower() != "haeoreum":
            self.logger.warning(
                f"Product source is not Haeoreum Gift: {product.source}"
            )
            # 해오름 소스로 설정
            product.source = "haeoreum"
            self.logger.info(f"Reset product source to Haeoreum Gift")

        processing_result = ProcessingResult(source_product=product)

        # 텍스트 유사도 임계값 설정
        text_threshold = self.config["MATCHING"].get("TEXT_SIMILARITY_THRESHOLD", 0.65)
        # 텍스트 초기 필터링을 위한 낮은 임계값 (성능 개선을 위한 필터링용)
        initial_text_threshold = text_threshold * 0.7  # 임계값의 70%

        # 고려기프트 매칭 검색
        try:
            self.logger.info(f"Searching Koryo Gift for: {product.name}")
            koryo_matches = self.koryo_scraper.search_product(product.name)

            if not koryo_matches:
                self.logger.info(
                    f"❌ 고려기프트에서 '{product.name}' 상품을 찾을 수 없음"
                )
            else:
                self.logger.debug(
                    f"✅ 고려기프트에서 '{product.name}' 상품 {len(koryo_matches)}개 발견"
                )

                # 1단계: 텍스트 유사도만 먼저 계산하여 후보군 추리기
                text_filtered_matches = []
                for match in koryo_matches:
                    text_sim = self.text_matcher.calculate_similarity(
                        product.name, match.name
                    )
                    if text_sim >= initial_text_threshold:
                        text_filtered_matches.append((match, text_sim))

                self.logger.info(
                    f"🔍 텍스트 유사도로 {len(text_filtered_matches)}/{len(koryo_matches)}개 후보 추려냄 (임계값: {initial_text_threshold:.2f})"
                )

                # 2단계: 텍스트 유사도가 높은 후보들에 대해서만 이미지 유사도 계산
                for match, text_sim in text_filtered_matches:
                    # 기본 MatchResult 생성
                    match_result = MatchResult(
                        source_product=product,
                        matched_product=match,
                        text_similarity=text_sim,
                        image_similarity=0.0,
                        combined_similarity=0.0,
                        price_difference=0.0,
                        price_difference_percent=0.0,
                    )

                    # 이미지 유사도 및 가격 차이 계산
                    self._calculate_image_similarity_and_price(match_result)

                    # 결과 추가
                    processing_result.koryo_matches.append(match_result)

            # 최적 매칭 찾기
            processing_result.best_koryo_match = self._find_best_match(
                processing_result.koryo_matches
            )

            if processing_result.best_koryo_match:
                if processing_result.best_koryo_match.image_similarity > 0:
                    self.logger.info(
                        f"✅ 고려기프트 매칭 (이미지 포함): {processing_result.best_koryo_match.matched_product.name}"
                    )
                else:
                    self.logger.info(
                        f"📝 고려기프트 매칭 (텍스트만): {processing_result.best_koryo_match.matched_product.name}"
                    )

        except Exception as e:
            self.logger.error(
                f"Error finding Koryo matches for {product.name}: {str(e)}",
                exc_info=True,
            )
            processing_result.error = f"Koryo search error: {str(e)}"

        # 네이버 매칭 검색
        try:
            self.logger.info(f"Searching Naver for: {product.name}")
            naver_matches = self._safe_naver_search(product.name)

            if not naver_matches:
                self.logger.info(f"❌ 네이버에서 '{product.name}' 상품을 찾을 수 없음")
            elif len(naver_matches) == 1 and getattr(naver_matches[0], 'id', '') == "no_match":
                self.logger.info(
                    f"❌ 네이버에서 '{product.name}' 상품을 찾을 수 없음 (no_match 반환)"
                )
            else:
                self.logger.debug(
                    f"✅ 네이버에서 '{product.name}' 상품 {len(naver_matches)}개 발견"
                )

                # 'no_match' 더미 상품 제외 - 이 부분의 필터링 로직도 수정
                real_matches = []
                for m in naver_matches:
                    if not hasattr(m, 'id') or m.id != "no_match":
                        real_matches.append(m)
                
                # 실제 검색 결과가 있는지 다시 확인
                if real_matches:
                    # 1단계: 텍스트 유사도만 먼저 계산하여 후보군 추리기
                    text_filtered_matches = []
                    for match in real_matches:
                        text_sim = self.text_matcher.calculate_similarity(
                            product.name, match.name
                        )
                        if text_sim >= initial_text_threshold:
                            text_filtered_matches.append((match, text_sim))

                    self.logger.info(
                        f"🔍 텍스트 유사도로 {len(text_filtered_matches)}/{len(real_matches)}개 후보 추려냄 (임계값: {initial_text_threshold:.2f})"
                    )

                    # 2단계: 텍스트 유사도가 높은 후보들에 대해서만 이미지 유사도 계산
                    for match, text_sim in text_filtered_matches:
                        # 기본 MatchResult 생성
                        match_result = MatchResult(
                            source_product=product,
                            matched_product=match,
                            text_similarity=text_sim,
                            image_similarity=0.0,
                            combined_similarity=0.0,
                            price_difference=0.0,
                            price_difference_percent=0.0,
                        )

                        # 이미지 유사도 및 가격 차이 계산
                        self._calculate_image_similarity_and_price(match_result)

                        # 결과 추가
                        processing_result.naver_matches.append(match_result)
                else:
                    self.logger.info(f"❌ 네이버에서 '{product.name}' 상품의 유효한 매치가 없음")

            # 최적 매칭 찾기
            processing_result.best_naver_match = self._find_best_match(
                processing_result.naver_matches
            )

            if processing_result.best_naver_match:
                if processing_result.best_naver_match.image_similarity > 0:
                    self.logger.info(
                        f"✅ 네이버 매칭 (이미지 포함): {processing_result.best_naver_match.matched_product.name}"
                    )
                else:
                    self.logger.info(
                        f"📝 네이버 매칭 (텍스트만): {processing_result.best_naver_match.matched_product.name}"
                    )

        except Exception as e:
            self.logger.error(
                f"Error finding Naver matches for {product.name}: {str(e)}",
                exc_info=True,
            )
            if not processing_result.error:  # 이전 오류가 없을 경우만 설정
                processing_result.error = f"Naver search error: {str(e)}"

        # 데이터 검증 및 비어있는 필드 처리
        self._ensure_valid_result(processing_result)

        return processing_result

    def _calculate_image_similarity_and_price(self, match_result: MatchResult) -> None:
        """
        이미지 유사도와 가격 차이를 계산하고 MatchResult에 설정

        Args:
            match_result: 업데이트할 MatchResult 객체
        """
        source_product = match_result.source_product
        matched_product = match_result.matched_product

        # 소스 이미지 URL 확인
        source_image_url = ""

        # 해오름 기프트(원본)의 이미지를 확인
        if source_product.source == "haeoreum":
            # 1. 본사 이미지 필드에서 먼저 확인
            if (
                "본사 이미지" in source_product.original_input_data
                and source_product.original_input_data["본사 이미지"]
            ):
                source_image_url = str(
                    source_product.original_input_data["본사 이미지"]
                ).strip()

            # 2. 이미지 URL 속성 확인
            if not source_image_url and source_product.image_url:
                source_image_url = source_product.image_url

            # 3. 본사상품링크 확인 (이미지가 없는 경우 직접 스크래핑하여 이미지 추출)
            if (
                not source_image_url
                and "본사상품링크" in source_product.original_input_data
                and source_product.original_input_data["본사상품링크"]
            ):
                product_link = str(
                    source_product.original_input_data["본사상품링크"]
                ).strip()
                if (
                    product_link
                    and "jclgift.com" in product_link
                    and "p_idx=" in product_link
                ):
                    self.logger.info(
                        f"No image URL found, scraping from product link: {product_link}"
                    )

                    try:
                        # URL에서 p_idx 파라미터 추출
                        import re

                        from ..scraping.haeoeum_scraper import HaeoeumScraper

                        p_idx_match = re.search(r"p_idx=(\d+)", product_link)
                        if p_idx_match:
                            p_idx = p_idx_match.group(1)

                            # 해오름 스크래퍼를 사용하여 이미지 추출
                            scraper = HaeoeumScraper(cache=self.cache)
                            scraped_product = scraper.get_product(p_idx)

                            if scraped_product and scraped_product.image_url:
                                source_image_url = scraped_product.image_url
                                self.logger.info(
                                    f"Successfully extracted image URL: {source_image_url}"
                                )

                                # 결과물을 소스 제품에 저장 (향후 사용)
                                source_product.image_url = source_image_url
                                source_product.original_input_data["본사 이미지"] = source_image_url

                                # 이미지 갤러리도 있으면 저장
                                if scraped_product.image_gallery:
                                    source_product.image_gallery = (
                                        scraped_product.image_gallery
                                    )
                                    self.logger.info(
                                        f"Added {len(scraped_product.image_gallery)} images to gallery"
                                    )
                            else:
                                self.logger.warning(
                                    f"Failed to extract image from {product_link}"
                                )
                    except Exception as e:
                        self.logger.error(
                            f"Error scraping image from product link: {str(e)}"
                        )
        else:
            # 다른 소스의 경우 단순히 image_url 속성 사용
            source_image_url = source_product.image_url

        # 매치된 이미지 URL 확인
        match_image_url = matched_product.image_url or ""

        # 매칭 소스 확인 (네이버 또는 고려기프트)
        match_source = (
            "네이버" if matched_product.source == "naver_api" else "고려기프트"
        )

        # 이미지 유사도 계산
        image_sim = 0.0  # 기본값

        # 양쪽 모두 이미지가 있는 경우: 이미지 유사도 계산
        if source_image_url and match_image_url:
            # 캐시 키 생성 (URL 해시 사용)
            import hashlib

            cache_key = f"img_sim_{hashlib.md5((source_image_url + match_image_url).encode()).hexdigest()}"

            # 캐시에서 유사도 값 확인
            cached_sim = self.cache.get(cache_key)
            if cached_sim is not None:
                image_sim = cached_sim
                self.logger.debug(f"🔄 캐시에서 이미지 유사도 로드: {image_sim:.2f}")
            else:
                self.logger.debug(
                    f"🖼️ 이미지 유사도 계산: {source_image_url} <-> {match_image_url}"
                )
                try:
                    # 이미지 해상도 축소 설정 (빠른 비교를 위함)
                    max_size = self.config["MATCHING"].get("MAX_IMAGE_DIMENSION", 256)

                    # 이미지 유사도 계산 시 해상도 제한
                    image_sim = self.image_matcher.calculate_similarity(
                        source_image_url, match_image_url, max_dimension=max_size
                    )

                    # 결과 캐싱 (1일 유지)
                    self.cache.set(cache_key, image_sim, ttl=86400)

                    self.logger.debug(f"  이미지 유사도 결과: {image_sim:.2f}")
                except Exception as e:
                    self.logger.warning(f"이미지 유사도 계산 중 오류: {str(e)}")
                    # 오류 발생 시 기본값 사용
                    image_sim = 0.0
        # 해오름 제품에 이미지가 없고 매칭된 제품에 이미지가 있는 경우
        elif not source_image_url and match_image_url:
            self.logger.warning(
                f"⚠️ 해오름 제품 '{source_product.name}'에 이미지가 없으나, {match_source}에 이미지 있음"
            )
            # 네이버는 이미지가 있을 확률이 높으므로 더 높은 기본값 부여
            if matched_product.source == "naver_api":
                image_sim = 0.5
            else:
                image_sim = 0.4
        # 매칭된 제품에 이미지가 없는 경우
        elif source_image_url and not match_image_url:
            self.logger.warning(
                f"⚠️ 해오름 제품 '{source_product.name}'에 이미지가 있으나, {match_source}에 이미지 없음"
            )
            # 텍스트 유사도가 높으면 이미지 유사도 기본값 부여 (0.3)
            if match_result.text_similarity >= 0.75:
                image_sim = 0.3
        # 둘 다 이미지가 없는 경우
        else:
            self.logger.warning(
                f"⚠️ 두 제품 모두 이미지 없음: 해오름 '{source_product.name}' <-> {match_source} '{matched_product.name}'"
            )
            # 텍스트 유사도가 매우 높은 경우만 약간의 기본값 부여
            if match_result.text_similarity >= 0.85:
                image_sim = 0.2

        # 이미지 유사도 설정
        match_result.image_similarity = image_sim

        # 통합 유사도 계산 및 설정
        match_result.combined_similarity = self.multimodal_matcher.calculate_similarity(
            match_result.text_similarity, match_result.image_similarity
        )

        # 가격 차이 계산
        price_diff = 0.0
        price_diff_percent = 0.0
        source_price = source_product.price

        if (
            source_price
            and source_price > 0
            and isinstance(matched_product.price, (int, float))
        ):
            price_diff = matched_product.price - source_price
            price_diff_percent = (
                (price_diff / source_price) * 100 if source_price != 0 else 0
            )

        # 가격 차이 설정
        match_result.price_difference = price_diff
        match_result.price_difference_percent = price_diff_percent

        self.logger.debug(
            f"  Match candidate {matched_product.name} ({matched_product.source}): "
            f"Txt={match_result.text_similarity:.2f}, Img={match_result.image_similarity:.2f}, "
            f"Comb={match_result.combined_similarity:.2f}, Price diff: {price_diff}"
        )

    def _ensure_valid_result(self, result: ProcessingResult) -> None:
        """결과 데이터가 유효한지 확인하고 필요한 기본값을 설정"""
        # 소스 제품 데이터 검증
        if (
            not hasattr(result.source_product, "original_input_data")
            or result.source_product.original_input_data is None
        ):
            result.source_product.original_input_data = {}

        # 필수 필드 존재 확인
        required_fields = [
            "구분",
            "담당자",
            "업체명",
            "업체코드",
            "상품Code",
            "중분류카테고리",
            "상품명",
            "기본수량(1)",
            "판매단가(V포함)",
            "본사상품링크",
        ]

        for field in required_fields:
            if field not in result.source_product.original_input_data:
                result.source_product.original_input_data[field] = ""

        # 중요 필드 유효성 검사
        if (
            "Code" in result.source_product.original_input_data
            and not result.source_product.original_input_data.get("상품Code")
        ):
            result.source_product.original_input_data["상품Code"] = (
                result.source_product.original_input_data["Code"]
            )

        if (
            "업체코드" in result.source_product.original_input_data
            and not result.source_product.original_input_data["업체코드"]
        ):
            if (
                hasattr(result.source_product, "product_code")
                and result.source_product.product_code
            ):
                result.source_product.original_input_data["업체코드"] = (
                    result.source_product.product_code
                )

    def _safe_naver_search(self, query: str) -> List[Product]:
        """안전하게 네이버 검색 실행"""
        products = []
        original_query = query

        # 검색 변형 시도들 (원래 검색어, 단어 수 줄이기)
        queries_to_try = []

        # 원래 검색어 추가
        queries_to_try.append(original_query)

        # 검색어에서 특수문자와 불필요한 정보 제거 (예: 규격, 수량 정보)
        cleaned_query = re.sub(r"[^\w\s]", " ", original_query)
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
        if cleaned_query != original_query:
            queries_to_try.append(cleaned_query)

        # 단어 수 줄이기 (긴 검색어의 경우 앞쪽 3-4 단어만 사용)
        words = cleaned_query.split()
        if len(words) > 4:
            shortened_query = " ".join(words[:4])
            queries_to_try.append(shortened_query)
        elif len(words) > 3:
            shortened_query = " ".join(words[:3])
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
                    self.logger.info(
                        f"✅ 검색 성공: '{current_query}'에서 {len(current_products)}개 상품 발견"
                    )

                    # 충분한 결과를 찾았으면 더 이상 시도하지 않음
                    if len(products) >= 10:
                        break
                else:
                    self.logger.warning(f"⚠️ '{current_query}'에 대한 검색 결과 없음")

            except TypeError as e:
                # 인자 오류 시 대체 호출
                self.logger.warning(
                    f"Type error during Naver search: {str(e)}. Trying alternative method."
                )
                try:
                    # 직접 내부 검색 로직 호출
                    if hasattr(self.naver_crawler, "_search_product_logic"):
                        current_products = self.naver_crawler._search_product_logic(
                            current_query, 50, None
                        )
                        if current_products:
                            products.extend(current_products)
                            self.logger.info(
                                f"✅ 대체 검색 성공: '{current_query}'에서 {len(current_products)}개 상품 발견"
                            )
                    elif hasattr(self.naver_crawler, "_search_product_async"):
                        # 비동기 호출 처리
                        import asyncio

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            current_products = loop.run_until_complete(
                                self.naver_crawler._search_product_async(
                                    current_query, 50, None
                                )
                            )
                            if current_products:
                                products.extend(current_products)
                                self.logger.info(
                                    f"✅ 비동기 검색 성공: '{current_query}'에서 {len(current_products)}개 상품 발견"
                                )
                        finally:
                            loop.close()
                except Exception as inner_e:
                    self.logger.error(
                        f"Alternative Naver search failed: {str(inner_e)}",
                        exc_info=True,
                    )

            except Exception as e:
                self.logger.error(f"Error in Naver search: {str(e)}", exc_info=True)

        # 중복 제거
        unique_products = {}
        for product in products:
            if product.id not in unique_products:
                unique_products[product.id] = product

        # 결과가 없거나 충분하지 않은 경우 다른 검색 엔진이나 방법을 시도할 수 있음
        if not unique_products:
            self.logger.warning(
                f"❌ 모든 검색 시도 후에도 '{original_query}'에 대한 검색 결과 없음"
            )
        else:
            self.logger.info(
                f"🎯 '{original_query}'에 대해 총 {len(unique_products)}개의 고유 상품 발견"
            )

        return list(unique_products.values())

    def _find_best_match(self, matches: List[MatchResult]) -> Optional[MatchResult]:
        """
        매칭 결과 중 최적의 결과 선택

        매뉴얼 요구사항:
        1. 상품 이름으로 검색하여 동일 상품 찾기
        2. 이미지로 제품 비교 (이미지 비교가 어려운 경우 규격 확인)
        3. 동일 상품으로 판단되면 가장 낮은 가격의 상품 선택
        """
        if not matches:
            self.logger.warning(
                "🚫 검색 결과 없음: 해당 상품이 고려기프트/네이버에 존재하지 않음"
            )
            return None

        # 이미지가 있는 매칭과 없는 매칭 분리
        matches_with_image = [m for m in matches if m.image_similarity > 0]
        matches_without_image = [m for m in matches if m.image_similarity == 0]

        # 로깅
        if not matches_with_image and matches_without_image:
            source_name = matches[0].source_product.name
            match_source = (
                "네이버"
                if matches[0].matched_product.source == "naver_api"
                else "고려기프트"
            )
            self.logger.info(
                f"📋 '{source_name}': {match_source}에서 {len(matches)} 매칭이 발견되었으나 이미지를 가진 매칭 없음"
            )

        # 임계값 설정
        text_threshold = self.config["MATCHING"].get("TEXT_SIMILARITY_THRESHOLD", 0.65)
        image_threshold = self.config["MATCHING"].get("IMAGE_SIMILARITY_THRESHOLD", 0.3)

        # 엄격한 매칭: 임계값 이상인 매칭만 필터링 (동일 상품으로 간주)
        valid_matches = [
            m
            for m in matches
            if m.text_similarity >= text_threshold
            and (
                # 이미지가 있는 경우는 이미지 유사도 검사
                (m.image_similarity >= image_threshold)
                or
                # 이미지가 없지만 텍스트 유사도가 매우 높은 경우(0.85 이상) 예외 허용
                (m.image_similarity == 0 and m.text_similarity >= 0.85)
            )
        ]

        # 임계값을 통과한 매칭이 있으면 그 중에서 최저가 선택
        if valid_matches:
            # 이미지 있는 매칭과 없는 매칭 중 선택 우선순위 결정
            valid_with_image = [
                m for m in valid_matches if m.image_similarity >= image_threshold
            ]

            if valid_with_image:
                # 이미지가 있는 매칭 중에서 최저가 선택
                best_match = min(
                    valid_with_image,
                    key=lambda x: (
                        x.matched_product.price
                        if x.matched_product.price > 0
                        else float("inf")
                    ),
                )
                self.logger.info(
                    f"💯 이미지 매칭 성공: {best_match.matched_product.name} (텍스트 유사도: {best_match.text_similarity:.2f}, 이미지 유사도: {best_match.image_similarity:.2f}, 가격: {best_match.matched_product.price})"
                )
            else:
                # 이미지 없이 텍스트만 매칭된 경우
                best_match = min(
                    valid_matches,
                    key=lambda x: (
                        x.matched_product.price
                        if x.matched_product.price > 0
                        else float("inf")
                    ),
                )
                self.logger.info(
                    f"📝 텍스트만 매칭 성공: {best_match.matched_product.name} (텍스트 유사도: {best_match.text_similarity:.2f}, 가격: {best_match.matched_product.price})"
                )
            return best_match

        # 임계값을 통과한 매칭이 없으면 더 낮은 임계값 시도
        relaxed_text_threshold = text_threshold * 0.75  # 25% 낮은 임계값
        relaxed_matches = [
            m for m in matches if m.text_similarity >= relaxed_text_threshold
        ]

        if relaxed_matches:
            # 낮은 임계값에서는 텍스트 유사도와 가격을 동시에 고려해 가장 적합한 매칭 선택
            # 텍스트 유사도로 정렬 후 상위 3개 중에서 최저가 선택
            top_matches = sorted(
                relaxed_matches, key=lambda x: x.text_similarity, reverse=True
            )[:3]
            best_match = min(
                top_matches,
                key=lambda x: (
                    x.matched_product.price
                    if x.matched_product.price > 0
                    else float("inf")
                ),
            )

            # 이미지 유무에 따른 로깅
            if best_match.image_similarity > 0:
                self.logger.warning(
                    f"⚠️ 낮은 임계값으로 이미지 매칭: {best_match.matched_product.name} (텍스트 유사도: {best_match.text_similarity:.2f}, 이미지 유사도: {best_match.image_similarity:.2f}, 가격: {best_match.matched_product.price})"
                )
            else:
                self.logger.warning(
                    f"⚠️ 낮은 임계값으로 텍스트만 매칭: {best_match.matched_product.name} (텍스트 유사도: {best_match.text_similarity:.2f}, 가격: {best_match.matched_product.price})"
                )
            return best_match

        # 모든 임계값에서 매칭을 찾지 못한 경우, 모든 매칭 중 가장 유사한 하나 반환 (매우 유연한 대안)
        if matches:
            # 통합 유사도로 정렬해 가장 높은 하나 선택
            best_match = max(matches, key=lambda x: x.combined_similarity)
            self.logger.warning(
                f"❗ 모든 임계값 실패, 가장 유사한 제품 선택: {best_match.matched_product.name} (통합 유사도: {best_match.combined_similarity:.2f})"
            )
            return best_match

        return None

    def process_files(
        self, input_files: List[str], output_dir: str = None, limit: int = None
    ) -> List[str]:
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
                        output_file = self._process_limited_file(
                            input_file, output_dir, limit
                        )
                    else:
                        output_file = self._process_single_file(input_file, output_dir)

                    if output_file:
                        output_files.append(output_file)

                except Exception as e:
                    self.logger.error(
                        f"파일 처리 중 오류 발생: {input_file}, 오류: {str(e)}",
                        exc_info=True,
                    )
                    continue

            return output_files

        except Exception as e:
            self.logger.error(f"파일 처리 중 오류 발생: {str(e)}", exc_info=True)
            return []

    def _process_limited_file(
        self, input_file: str, output_dir: str = None, limit: int = 10
    ) -> Optional[str]:
        """제한된 수의 상품만 처리합니다."""
        # Reset running flag at the start of processing a file
        self._is_running = True
        try:
            start_time = datetime.now()
            self.logger.info(f"파일 처리 시작: {input_file} (최대 {limit}개 상품)")

            # Excel 파일 읽기
            df = pd.read_excel(input_file)
            if df.empty:
                self.logger.error(f"파일이 비어있습니다: {input_file}")
                return None

            # 데이터 정제
            # Ensure data cleaning uses the configured cleaner
            df = self.data_cleaner.clean_dataframe(df)

            # 제한된 수의 상품만 선택
            if len(df) > limit:
                self.logger.info(f"전체 {len(df)}개 중 {limit}개 상품만 처리합니다.")
                df = df.head(limit)

            # 결과 저장을 위한 리스트
            results = []
            total_items = len(df) # Get total items *after* limiting

            # Emit initial progress
            if self.progress_callback:
                 try:
                    self.progress_callback(0, total_items)
                 except Exception as cb_e:
                    self.logger.error(f"Error in initial progress callback: {cb_e}")

            # 각 상품 처리 (using ThreadPoolExecutor for potential future parallelization within limit)
            # Create futures list
            futures = []
            product_map = {} # To map future back to product if needed

            for idx, row in df.iterrows():
                 # Check if thread is still running before processing
                if not self._is_running:
                     self.logger.warning("Processing stopped by user request (limited file).")
                     break # Exit loop if stopped

                try:
                    # Create Product object
                    product = self.product_factory.create_product_from_row(row)
                    if not product:
                        self.logger.warning(f"Could not create product from row {idx+1}. Skipping.")
                        # Update progress even for skipped items
                        if self.progress_callback:
                            try:
                                self.progress_callback(idx + 1, total_items)
                            except Exception as cb_e:
                                self.logger.error(f"Error in progress callback (skipped item): {cb_e}")
                        continue

                    # Submit processing task
                    future = self.executor.submit(self._process_single_product, product)
                    futures.append(future)
                    product_map[future] = product # Store product associated with future

                except Exception as e:
                    self.logger.error(
                        f"상품 생성/제출 중 오류 발생 (행 {idx+1}): {str(e)}", exc_info=True
                    )
                    # Emit progress even on error to keep UI updated
                    if self.progress_callback:
                         try:
                             self.progress_callback(idx + 1, total_items)
                         except TypeError:
                             pass # Ignore signature mismatch error here too
                    continue

            # Process completed futures
            processed_count = 0
            from concurrent.futures import as_completed

            for future in as_completed(futures):
                 # Check if stopped before processing result
                 if not self._is_running:
                     self.logger.warning("Processing stopped during result collection (limited file).")
                     if not future.done():
                         future.cancel()
                     continue # Skip remaining futures

                 product = product_map.get(future) # Get the original product
                 processed_count += 1 # Increment counter for each future completed/checked

                 try:
                    result = future.result(timeout=300) # Get result
                    if result:
                        results.append(result)

                 except Exception as e:
                    error_msg = f"상품 처리 중 오류 발생 (Product ID: {product.id if product else 'N/A'}): {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    # Optionally add an error result if needed
                    if product:
                         results.append(ProcessingResult(source_product=product, error=str(e)))

                 finally:
                    # --- Call progress callback ---
                    if self.progress_callback:
                        # Use try-except block to avoid crashing if callback signature mismatches
                        try:
                             # Use processed_count which reflects completed futures
                             self.progress_callback(processed_count, total_items)
                        except Exception as cb_e:
                             self.logger.error(f"Error in progress callback (limited file): {cb_e}")

            # If processing was stopped, skip output generation
            if not self._is_running:
                self.logger.warning("Processing was stopped. Skipping output file generation (limited file).")
                return None

            # 결과가 있는 경우에만 출력 파일 생성
            if results:
                # 출력 파일명 생성
                output_file = self.excel_manager.generate_enhanced_output(
                    results, input_file, output_dir # Pass output_dir
                )

                # 후처리 작업 수행 (하이퍼링크, 필터링 등)
                output_file = self.post_process_output_file(output_file)

                # 처리 완료 로깅
                end_time = datetime.now()
                processing_time = end_time - start_time
                self.logger.info(
                    f"Limited processing finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.logger.info(f"Total processing time (limited): {processing_time}")

                self.logger.info(f"파일 처리 완료: {output_file}")
                return output_file

            else:
                self.logger.info("No results generated from limited processing.")
                return None

        except Exception as e:
            self.logger.error(f"파일 처리 중 오류 발생: {str(e)}", exc_info=True)
            return None

    def process_koryo_products(self, search_query: str, output_path: str, max_items: int = 50):
        """Process Koryo Gift products and save to Excel"""
        try:
            self.logger.info(f"Starting Koryo Gift product processing for query: {search_query}")
            
            # 제품 검색 및 데이터 수집
            products = self.koryo_scraper.search_product(search_query, max_items=max_items)
            
            if not products:
                self.logger.warning(f"No products found for query: {search_query}")
                # 빈 결과를 저장할 때도 헤더와 상태 메시지 포함
                self.excel_manager.save_products([], output_path, "검색결과없음")
                return
            
            self.logger.info(f"Found {len(products)} products from Koryo Gift")
            
            # 검색 결과를 엑셀로 저장
            sheet_name = f"koryo_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.excel_manager.save_products(products, output_path, sheet_name)
            
            self.logger.info(f"Successfully saved Koryo Gift products to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing Koryo Gift products: {str(e)}", exc_info=True)
            raise

    def process_search_results(self, search_query: str, output_path: str, max_items: int = 50):
        """Process search results from multiple sources"""
        try:
            all_products = []
            
            # 고려기프트 제품 검색
            try:
                koryo_products = self.koryo_scraper.search_product(search_query, max_items=max_items)
                if koryo_products:
                    self.logger.info(f"Found {len(koryo_products)} products from Koryo Gift")
                    all_products.extend(koryo_products)
                else:
                    self.logger.warning("No products found from Koryo Gift")
            except Exception as e:
                self.logger.error(f"Error searching Koryo Gift: {str(e)}")
            
            # 네이버 제품 검색
            try:
                naver_products = self.naver_crawler.search_product(search_query, max_items=max_items)
                if naver_products:
                    self.logger.info(f"Found {len(naver_products)} products from Naver")
                    all_products.extend(naver_products)
                    
                    # 네이버 검색 결과만 별도로 저장
                    naver_sheet_name = f"naver_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    naver_output_path = output_path.replace('.xlsx', '_naver.xlsx')
                    self.excel_manager.save_products(naver_products, naver_output_path, naver_sheet_name)
                    self.logger.info(f"Successfully saved Naver products to: {naver_output_path}")
                else:
                    self.logger.warning("No products found from Naver")
            except Exception as e:
                self.logger.error(f"Error searching Naver: {str(e)}")
            
            # 다른 소스의 제품 검색 로직...
            
            if not all_products:
                self.logger.warning(f"No products found for query: {search_query}")
                # 빈 결과를 저장할 때도 헤더와 상태 메시지 포함
                self.excel_manager.save_products([], output_path, "검색결과없음")
                return
            
            # 모든 검색 결과를 엑셀로 저장
            sheet_name = f"search_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.excel_manager.save_products(all_products, output_path, sheet_name)
            
            self.logger.info(f"Successfully saved all search results to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing search results: {str(e)}", exc_info=True)
            raise

    def validate_product_data(self, product: Product) -> bool:
        """Validate product data before saving"""
        if not product:
            return False
            
        # 필수 필드 검증
        required_fields = {
            'name': product.name,
            'price': product.price,
            'url': product.url,
            'source': product.source
        }
        
        for field, value in required_fields.items():
            if not value:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        # 이미지 URL 검증
        if not product.image_url and not product.image_gallery:
            self.logger.warning("No images found for product")
            return False
            
        return True
