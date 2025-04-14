import logging
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.caching import FileCache
from utils.preprocessing import send_report_email

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
            enable_compression=self.config["PROCESSING"].get("ENABLE_COMPRESSION", True),  # 압축 기본 활성화
            compression_level=self.config["PROCESSING"].get("COMPRESSION_LEVEL", 1),  # 낮은 압축 레벨로 속도 향상
        )

        # 매칭 컴포넌트 초기화 - 메모리 사용량 감소 및 속도 향상을 위한 최적화
        self.text_matcher = TextMatcher(
            cache=self.cache,
            use_stemming=self.config["MATCHING"].get("USE_STEMMING", False),  # 성능 향상을 위해 스테밍 비활성화
            similarity_threshold=self.config["MATCHING"].get("TEXT_SIMILARITY_THRESHOLD", 0.7)
        )

        # 이미지 처리 최대 해상도 설정
        max_image_dimension = self.config["MATCHING"].get("MAX_IMAGE_DIMENSION", 128)  # 기본값 낮춤 (256 → 128)
        self.logger.info(f"이미지 처리 최대 해상도: {max_image_dimension}px")

        self.image_matcher = ImageMatcher(
            cache=self.cache,
            similarity_threshold=self.config["MATCHING"]["IMAGE_SIMILARITY_THRESHOLD"],
            max_image_dimension=max_image_dimension
        )

        self.multimodal_matcher = MultiModalMatcher(
            text_weight=self.config["MATCHING"]["TEXT_WEIGHT"],
            image_weight=self.config["MATCHING"]["IMAGE_WEIGHT"],
            text_matcher=self.text_matcher,
            image_matcher=self.image_matcher,
            similarity_threshold=self.config["MATCHING"].get("TEXT_SIMILARITY_THRESHOLD", 0.75)
        )

        # 스크래퍼 초기화 - 공통 설정
        scraper_config = {
            "max_retries": min(3, self.config["PROCESSING"].get("MAX_RETRIES", 3)),  # 최대 3회로 제한
            "cache": self.cache,
            "timeout": min(15, self.config["PROCESSING"].get("REQUEST_TIMEOUT", 15)),  # 최대 15초로 제한
            "connect_timeout": 5,  # 연결 타임아웃 추가
            "read_timeout": 10,  # 읽기 타임아웃 추가
        }

        # 해오름 스크래퍼
        self.haeoeum_scraper = HaeoeumScraper(**scraper_config)

        # 고려 스크래퍼
        self.koryo_scraper = KoryoScraper(**scraper_config)

        # 네이버 크롤러
        naver_config = {
            "max_retries": min(3, self.config["PROCESSING"].get("MAX_RETRIES", 3)),  # 최대 3회로 제한
            "cache": self.cache,
            "timeout": min(15, self.config["PROCESSING"].get("REQUEST_TIMEOUT", 15)),  # 최대 15초로 제한
        }
        
        # 프록시 사용 여부 확인
        use_proxies = False
        if "NETWORK" in self.config and self.config["NETWORK"].get("USE_PROXIES") == "True":
            use_proxies = True
            self.logger.info("프록시 사용 모드로 네이버 크롤러를 초기화합니다.")
            naver_config["use_proxies"] = True

        self.naver_crawler = NaverShoppingCrawler(**naver_config)

        # 스크래퍼 설정 적용
        if "SCRAPING" in self.config:
            self._configure_scrapers(self.config["SCRAPING"])

        # 시스템 코어 수 기반 최적화된 병렬 처리 설정
        import multiprocessing
        import psutil  # 메모리 사용량 확인용

        available_memory = psutil.virtual_memory().available // (1024 * 1024)  # MB 단위
        self.logger.info(f"사용 가능한 메모리: {available_memory}MB")

        cpu_count = multiprocessing.cpu_count()
        
        # 메모리와 CPU 코어 수를 고려하여 워커 수 결정
        if available_memory < 1024:  # 1GB 미만
            max_workers = max(2, min(cpu_count // 2, 8))
        elif available_memory < 4096:  # 4GB 미만
            max_workers = max(4, min(cpu_count, 12))
        else:  # 4GB 이상
            max_workers = max(6, min(cpu_count * 2, 16))
            
        self.logger.info(f"병렬 처리 워커 수: {max_workers} (CPU 코어: {cpu_count})")
        
        # 스레드풀 초기화
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix="ProductProcessor"
        )

        # 배치 크기 최적화 - 메모리 사용량 고려
        if available_memory < 1024:  # 1GB 미만
            default_batch = max(2, min(5, cpu_count // 2))
        elif available_memory < 4096:  # 4GB 미만
            default_batch = max(5, min(10, cpu_count))
        else:  # 4GB 이상
            default_batch = max(10, min(20, cpu_count * 2))
            
        self.batch_size = self.config["PROCESSING"].get("BATCH_SIZE", default_batch)
        self.logger.info(f"배치 크기 설정: {self.batch_size}")

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
            # 0. 기본 엑셀 후처리 (@ 기호 제거, 이미지 수식 개선 등)
            processed_file = self.excel_manager.post_process_excel_file(output_file)
            
            # 1. 하이퍼링크 추가
            linked_file = self.excel_manager.add_hyperlinks_to_excel(processed_file)

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
    ) -> Tuple[Optional[str], Optional[str]]:
        """단일 파일 처리 로직 (체크포인트 지원)"""
        # 체크포인트 파일 경로 생성
        input_name = os.path.basename(input_file)
        base_name, _ = os.path.splitext(input_name)
        checkpoint_dir = os.path.join(os.path.dirname(input_file), '.checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_checkpoint.json")
        
        # 체크포인트 확인
        checkpoint_data = self._load_checkpoint(checkpoint_file)
        if checkpoint_data and 'status' in checkpoint_data and checkpoint_data['status'] == 'complete':
            self.logger.info(f"이미 완료된 파일입니다: {input_file}, 체크포인트에서 결과 로드")
            return checkpoint_data.get('intermediate_file'), checkpoint_data.get('output_file')
            
        try:
            # 체크포인트 진행 상태 기록
            self._update_checkpoint(checkpoint_file, {
                'status': 'processing',
                'input_file': input_file,
                'start_time': datetime.now().isoformat(),
                'step': 'started'
            })
            
            # 출력 디렉토리 설정
            if not output_dir:
                output_dir = self.config["PATHS"]["OUTPUT_DIR"]
            os.makedirs(output_dir, exist_ok=True)
            
            # 중간 파일 경로 설정
            intermediate_dir = os.path.join(output_dir, "intermediate")
            os.makedirs(intermediate_dir, exist_ok=True)
            intermediate_file = os.path.join(intermediate_dir, f"{base_name}_intermediate.xlsx")
            
            # 최종 파일 경로 설정
            final_dir = os.path.join(output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            output_file = os.path.join(final_dir, f"{base_name}_final.xlsx")
            
            # 파일 처리 로직
            # 1. 엑셀 전처리
            processed_file = self.process_excel_functionality(input_file)
            
            # 2. 중간 파일 생성
            self.excel_manager.save_products([], intermediate_file)
            
            # 3. 후처리 및 최종 파일 생성
            final_file = self.post_process_output_file(intermediate_file)
            if final_file != intermediate_file:
                os.rename(final_file, output_file)
            
            # 체크포인트 업데이트 - 처리 완료
            self._update_checkpoint(checkpoint_file, {
                'status': 'complete',
                'input_file': input_file,
                'intermediate_file': intermediate_file,
                'output_file': output_file,
                'end_time': datetime.now().isoformat()
            })
            
            return intermediate_file, output_file
            
        except Exception as e:
            # 오류 발생 시 체크포인트에 기록
            self._update_checkpoint(checkpoint_file, {
                'status': 'error',
                'input_file': input_file,
                'error': str(e),
                'error_time': datetime.now().isoformat()
            })
            self.logger.error(f"파일 처리 중 오류 발생: {input_file}, {str(e)}")
            raise

    def _load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """체크포인트 파일 로드"""
        if not os.path.exists(checkpoint_file):
            return None
            
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.loads(f.read())
        except Exception as e:
            self.logger.warning(f"체크포인트 파일 로드 실패: {checkpoint_file}, {str(e)}")
            return None
            
    def _update_checkpoint(self, checkpoint_file: str, data: Dict) -> bool:
        """체크포인트 파일 업데이트"""
        try:
            # 기존 데이터가 있으면 병합
            existing_data = self._load_checkpoint(checkpoint_file) or {}
            existing_data.update(data)
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            self.logger.warning(f"체크포인트 파일 업데이트 실패: {checkpoint_file}, {str(e)}")
            return False

    def run_workflow_from_manual(self, input_file: str, output_dir: str = None, 
                            send_email: bool = True, generate_second_stage: bool = True,
                            email_recipient: str = 'dasomas@kakao.com',
                            resume_from_checkpoint: bool = True):
        """
        작업메뉴얼에 따른 전체 워크플로우 실행
        
        1. 상품명 전처리 (1- 및 특수문자 제거)
        2. 네이버쇼핑 및 고려기프트 검색
        3. 1차 결과 생성 및 이메일 전송
        4. 2차 결과 생성 (필요시)
        
        Args:
            input_file: 입력 파일 경로
            output_dir: 출력 디렉토리 (기본값: 설정파일 기준)
            send_email: 이메일 전송 여부
            generate_second_stage: 2차 파일 생성 여부
            email_recipient: 이메일 수신자
            resume_from_checkpoint: 체크포인트에서 재개 여부
            
        Returns:
            dict: 처리 결과 요약
        """
        # 모듈 함수 직접 참조 대신 필요 시점에 임포트 (순환 참조 방지)
        from utils.preprocessing import send_report_email
        import importlib
        import json
        
        start_time = datetime.now()
        self.logger.info(f"작업메뉴얼 워크플로우 시작: {input_file}")
        
        # 체크포인트 파일 경로 생성
        input_name = os.path.basename(input_file)
        base_name, _ = os.path.splitext(input_name)
        checkpoint_dir = os.path.join(os.path.dirname(input_file), '.checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_workflow_checkpoint.json")
        
        # 체크포인트에서 재개 시도
        if resume_from_checkpoint:
            checkpoint_data = self._load_checkpoint(checkpoint_file)
            if checkpoint_data and checkpoint_data.get('status') == 'complete':
                self.logger.info(f"이미 완료된 워크플로우입니다: {input_file}")
                
                # 완료된 작업 결과 반환
                return {
                    "input_file": input_file,
                    "first_stage_file": checkpoint_data.get('first_stage_file'),
                    "second_stage_file": checkpoint_data.get('second_stage_file'),
                    "email_sent": checkpoint_data.get('email_sent', False),
                    "email_recipient": checkpoint_data.get('email_recipient'),
                    "start_time": checkpoint_data.get('start_time'),
                    "end_time": checkpoint_data.get('end_time'),
                    "duration_seconds": checkpoint_data.get('duration_seconds', 0),
                    "status": "success",
                    "resumed_from_checkpoint": True
                }
                
            elif checkpoint_data and checkpoint_data.get('status') == 'processing':
                # 처리 중인 작업 재개
                self.logger.info(f"중단된 워크플로우를 재개합니다: {input_file}")
                # 진행 상태에 따라 단계별 재개 가능
                first_stage_file = checkpoint_data.get('first_stage_file')
                second_stage_file = checkpoint_data.get('second_stage_file')
                email_sent = checkpoint_data.get('email_sent', False)
                
                # 체크포인트 상태 업데이트
                self._update_checkpoint(checkpoint_file, {
                    'resumed_at': datetime.now().isoformat()
                })
            else:
                # 새 체크포인트 시작
                self._update_checkpoint(checkpoint_file, {
                    'status': 'processing',
                    'input_file': input_file,
                    'start_time': start_time.isoformat(),
                    'step': 'started'
                })
                first_stage_file = None
                second_stage_file = None
                email_sent = False
        else:
            # 체크포인트 무시하고 새로 시작
            self._update_checkpoint(checkpoint_file, {
                'status': 'processing',
                'input_file': input_file,
                'start_time': start_time.isoformat(),
                'step': 'started'
            })
            first_stage_file = None
            second_stage_file = None
            email_sent = False
        
        try:
            # 출력 디렉토리 확인
            if not output_dir:
                output_dir = self.config["PATHS"]["OUTPUT_DIR"]
                os.makedirs(output_dir, exist_ok=True)
                
            # 1. 파일 처리 (1차 결과)
            if not first_stage_file:
                # 진행상황 콜백 업데이트
                if self.progress_callback:
                    self.progress_callback("1차 파일 처리 중...", 25)
                    
                # process_excel_file 함수는 없으므로 _process_single_file 사용
                _, first_stage_file = self._process_single_file(input_file, output_dir)
                
                if not first_stage_file:
                    self.logger.error(f"1차 파일 생성 실패: {input_file}")
                    raise ValueError(f"입력 파일 처리 중 오류 발생: {input_file}")
                    
                self.logger.info(f"1차 파일 생성 완료: {first_stage_file}")
                
                # 체크포인트 업데이트
                self._update_checkpoint(checkpoint_file, {
                    'step': 'first_stage_complete',
                    'first_stage_file': first_stage_file
                })
            
            # 2. 이메일 전송
            if not email_sent and send_email and first_stage_file:
                # 진행상황 콜백 업데이트
                if self.progress_callback:
                    self.progress_callback("이메일 전송 중...", 50)
                    
                email_sent = send_report_email(first_stage_file, recipient_email=email_recipient)
                if email_sent:
                    self.logger.info(f"1차 결과 이메일 전송 완료: {email_recipient}")
                    
                    # 체크포인트 업데이트
                    self._update_checkpoint(checkpoint_file, {
                        'step': 'email_sent',
                        'email_sent': True,
                        'email_recipient': email_recipient
                    })
                else:
                    self.logger.warning("1차 결과 이메일 전송 실패")
            
            # 3. 2차 파일 생성 (필요시)
            if not second_stage_file and generate_second_stage and first_stage_file:
                # 진행상황 콜백 업데이트
                if self.progress_callback:
                    self.progress_callback("2차 파일 생성 중...", 75)
                    
                # 동적으로 process_excel 모듈 로드
                process_excel = importlib.import_module('process_excel')
                second_stage_file = process_excel.process_first_to_second_stage(
                    first_stage_file, output_dir
                )
                self.logger.info(f"2차 파일 생성 완료: {second_stage_file}")
                
                # 체크포인트 업데이트
                self._update_checkpoint(checkpoint_file, {
                    'step': 'second_stage_complete',
                    'second_stage_file': second_stage_file
                })
            
            # 처리 완료 시간
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 진행상황 콜백 업데이트
            if self.progress_callback:
                self.progress_callback("워크플로우 완료", 100)
            
            # 체크포인트 업데이트 - 완료 상태
            self._update_checkpoint(checkpoint_file, {
                'status': 'complete',
                'end_time': end_time.isoformat(),
                'duration_seconds': duration
            })
            
            # 결과 요약
            result = {
                "input_file": input_file,
                "first_stage_file": first_stage_file,
                "second_stage_file": second_stage_file,
                "email_sent": email_sent,
                "email_recipient": email_recipient if email_sent else None,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "success"
            }
            
            self.logger.info(f"작업메뉴얼 워크플로우 완료: {duration:.1f}초 소요")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"작업메뉴얼 워크플로우 실패: {str(e)}", exc_info=True)
            
            # 체크포인트 업데이트 - 오류 상태
            self._update_checkpoint(checkpoint_file, {
                'status': 'error',
                'error': str(e),
                'error_time': end_time.isoformat(),
                'duration_seconds': duration
            })
            
            # 진행상황 콜백 업데이트 (오류)
            if self.progress_callback:
                self.progress_callback(f"오류 발생: {str(e)}", -1)
            
            # 오류 요약
            result = {
                "input_file": input_file,
                "first_stage_file": first_stage_file if 'first_stage_file' in locals() else None,
                "second_stage_file": second_stage_file if 'second_stage_file' in locals() else None,
                "email_sent": email_sent if 'email_sent' in locals() else False,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "error",
                "error": str(e)
            }
            
            return result
