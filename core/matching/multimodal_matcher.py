import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
import time
from configparser import ConfigParser
from logging import Logger

import numpy as np

from utils.caching import FileCache, cache_result
from .base_matcher import BaseMatcher

from ..data_models import MatchResult, Product
from .image_matcher import ImageMatcher
from .text_matcher import TextMatcher


class MultiModalMatcher(BaseMatcher):
    """
    Multimodal product matcher that combines text and image similarity.

    업데이트된 기능:
    - 매뉴얼 요구사항에 맞게 매칭 로직 강화
    - 이미지와 규격이 동일한 경우만 매칭으로 판단
    - 가격차이 계산 및 필터링 로직 추가
    - 적응형 가중치 조정 시스템 추가
    - 카테고리 기반 필터링 추가
    - 유사도 점수 향상을 위한 보너스 시스템 구현
    - SIFT+RANSAC 및 AKAZE 기반 고도화된 이미지 매칭
    - KoSBERT 한국어 특화 텍스트 매칭
    - 고급 제품명 토큰화 기능 추가
    - 가격 필터링 로직 개선
    """

    def __init__(
        self,
        text_weight: float = 0.7,
        image_weight: float = 0.3,
        text_matcher: Optional[TextMatcher] = None,
        image_matcher: Optional[ImageMatcher] = None,
        similarity_threshold: float = 0.75,
        config: Optional[Dict] = None,
        logger: Optional[Logger] = None
    ):
        """
        멀티모달 매처 초기화
        
        Args:
            text_weight: 텍스트 매칭 가중치 (0-1)
            image_weight: 이미지 매칭 가중치 (0-1)
            text_matcher: 텍스트 매처 인스턴스
            image_matcher: 이미지 매처 인스턴스
            similarity_threshold: 유사도 임계값
            config: 설정 딕셔너리
            logger: 로거 객체
        """
        super().__init__(config, logger)
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # 매칭 설정 로드
        self.matching_settings = self.config.get("MULTIMODAL_MATCHER", {})
        
        # 기본 매칭 설정
        self.text_weight = float(self.matching_settings.get("text_weight", text_weight))
        self.image_weight = float(self.matching_settings.get("image_weight", image_weight))
        self.similarity_threshold = float(self.matching_settings.get("similarity_threshold", similarity_threshold))
        self.text_similarity_threshold = float(self.matching_settings.get("text_similarity_threshold", similarity_threshold))
        self.image_similarity_threshold = float(self.matching_settings.get("image_similarity_threshold", similarity_threshold))
            
        # 가격 설정
        self.excel_settings = self.config.get("EXCEL", {})
        self.price_min = float(self.excel_settings.get("price_min", 0))
        self.price_max = float(self.excel_settings.get("price_max", 10000000000))
        self.price_diff_threshold = float(self.matching_settings.get("price_diff_threshold", 0.3))
            
        # 매칭 처리 설정
        self.processing_settings = self.config.get("PROCESSING", {})
        self.batch_size = int(self.processing_settings.get("batch_size", 5))
        self.max_retries = int(self.processing_settings.get("max_retries", 5))
        self.max_workers = int(self.processing_settings.get("max_workers", 4))
        self.cache_duration = int(self.processing_settings.get("cache_duration", 86400))
        self.auto_clean_product_names = bool(self.processing_settings.get("auto_clean_product_names", False))
        self.exponential_backoff = bool(self.processing_settings.get("exponential_backoff", True))
            
        # 스크래핑 설정
        self.scraping_settings = self.config.get("SCRAPING", {})
        self.extraction_timeout = int(self.scraping_settings.get("extraction_timeout", 30))
        self.politeness_delay = int(self.scraping_settings.get("politeness_delay", 2000))
            
        # 가중치 합이 1이 되도록 정규화
        total_weight = self.text_weight + self.image_weight
        if total_weight != 1.0:
            self.logger.warning(f"가중치 합이 1이 아닙니다. 정규화합니다. (현재 합: {total_weight})")
            self.text_weight /= total_weight
            self.image_weight /= total_weight
        
        # 캐시 객체 초기화 (캐시 공유)
        self.cache = self.config.get("cache", None)
        if self.config.get("create_cache", False) and not self.cache:
            try:
                self.cache = FileCache(
                    cache_dir=self.config.get("cache_dir", ".cache"),
                    ttl=self.cache_duration
                )
                self.logger.info(f"Created FileCache with TTL {self.cache_duration}s")
            except Exception as e:
                self.logger.warning(f"Failed to create FileCache: {e}")
                self.cache = None
            
        # 텍스트 및 이미지 매처 초기화
        if text_matcher:
            self.text_matcher = text_matcher
        else:
            try:
                self.text_matcher = TextMatcher(self.config, self.logger)
                self.logger.info("TextMatcher initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize TextMatcher: {e}", exc_info=True)
                # Fallback to basic initialization
                self.text_matcher = TextMatcher({}, self.logger)
        
        if image_matcher:
            self.image_matcher = image_matcher
        else:
            try:
                self.image_matcher = ImageMatcher(self.config, self.logger)
                self.logger.info("ImageMatcher initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize ImageMatcher: {e}", exc_info=True)
                # Fallback to basic initialization
                self.image_matcher = ImageMatcher({}, self.logger)
        
        # 적응형 가중치 설정
        self.use_adaptive_weights = bool(self.matching_settings.get("use_adaptive_weights", True))
        self.min_text_weight = float(self.matching_settings.get("min_text_weight", 0.3))
        self.min_image_weight = float(self.matching_settings.get("min_image_weight", 0.2))

        # 카테고리 매핑 정의 (카테고리 호환성 매트릭스)
        self.category_compatibility = self.config.get("category_compatibility", {
            # 주요 카테고리: 완전 호환 카테고리 목록
            "전자제품": ["가전", "디지털", "컴퓨터", "IT기기", "휴대폰", "카메라"],
            "패션": ["의류", "신발", "가방", "액세서리", "쥬얼리"],
            "식품": ["식료품", "음료", "건강식품", "간식"],
            "뷰티": ["화장품", "헤어", "바디", "스킨케어"],
            "생활용품": ["주방", "욕실", "청소", "수납"],
        })

        # 필터링 대상 키워드
        self.filter_keywords = self.config.get("filter_keywords", [
            "판촉", "기프트", "답례품", "기념품", "인쇄", "각인", "제작", "호갱", "몽키", "홍보",
        ])

    def match(self, source_product: Product, target_product: Product) -> float:
        """Calculate similarity score between source and target products
        
        Args:
            source_product: Source product to match from
            target_product: Target product to match against
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Input validation
            if not source_product or not target_product:
                self.logger.warning("Invalid product input for matching")
                return 0.0
                
            # Calculate individual similarities
            text_similarity = self.text_matcher.match(source_product.name, target_product.name)
            
            # Check if images are available for image similarity
            has_images = bool(source_product.image_url and target_product.image_url)
            image_similarity = 0.0
            
            if has_images:
                image_similarity = self.image_matcher.match(source_product.image_url, target_product.image_url)
            
            # Calculate adaptive weights if enabled
            if self.use_adaptive_weights:
                text_weight, image_weight = self._calculate_adaptive_weights(
                    text_similarity, image_similarity, has_images
                )
            else:
                text_weight, image_weight = self.text_weight, self.image_weight
            
            # Calculate combined similarity
            combined_similarity = (text_similarity * text_weight) + (image_similarity * image_weight)
            
            # Apply threshold (optional - might be better to do in find_matches)
            # return combined_similarity if combined_similarity >= self.similarity_threshold else 0.0
            return combined_similarity
            
        except Exception as e:
            self.logger.error(f"Error in multimodal matching between {source_product.id} and {target_product.id}: {str(e)}", exc_info=True)
            return 0.0

    def _calculate_adaptive_weights(
        self, text_similarity: float, image_similarity: float, has_images: bool
    ) -> Tuple[float, float]:
        """텍스트와 이미지 유사도 신뢰도에 기반한 가중치 동적 계산"""
        # 기본 가중치 설정
        text_weight = self.text_weight
        image_weight = self.image_weight

        # 이미지가 없는 경우
        if not has_images:
            text_weight = 1.0
            image_weight = 0.0
            return text_weight, image_weight

        # 텍스트 유사도가 매우 높은 경우 (0.9 이상)
        if text_similarity > 0.9:
            text_weight *= 1.2
            image_weight *= 0.8
        # 텍스트 유사도가 낮은 경우 (0.3 이하)
        elif text_similarity < 0.3:
            text_weight *= 0.8
            image_weight *= 1.2

        # 이미지 유사도가 매우 높은 경우 (0.9 이상)
        if image_similarity > 0.9:
            image_weight *= 1.2
            text_weight *= 0.8
        # 이미지 유사도가 낮은 경우 (0.3 이하)
        elif image_similarity < 0.3:
            image_weight *= 0.8
            text_weight *= 1.2

        # 최소 가중치 보장
        text_weight = max(text_weight, self.min_text_weight)
        image_weight = max(image_weight, self.min_image_weight)

        # 가중치 합이 1이 되도록 정규화
        total = text_weight + image_weight
        text_weight = text_weight / total
        image_weight = image_weight / total

        return text_weight, image_weight

    def _check_category_compatibility(
        self, source_category: Optional[str], candidate_category: Optional[str]
    ) -> bool:
        """두 카테고리의 호환성 체크"""
        # 카테고리가 없으면 호환된다고 간주
        if not source_category or not candidate_category:
            return True

        try:
            # 카테고리 정규화
            source_category = source_category.lower().strip()
            candidate_category = candidate_category.lower().strip()

            # 정확히 일치하면 호환
            if source_category == candidate_category:
                return True

            # 카테고리 별칭 확인 (캐시 활용)
            if self.cache:
                cache_key = f"category_comp_{source_category}_{candidate_category}"
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

            # 호환성 매트릭스에서 확인
            for main_category, compatible_categories in self.category_compatibility.items():
                if (source_category in compatible_categories and 
                    candidate_category in compatible_categories):
                    if self.cache:
                        self.cache.set(cache_key, True, ttl=3600)  # 1시간 캐싱
                    return True

            # 호환되지 않음
            if self.cache:
                self.cache.set(cache_key, False, ttl=3600)
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking category compatibility: {e}", exc_info=True)
            # 오류 발생 시 호환된다고 간주 (false negative 방지)
            return True

    def _check_price_compatibility(self, source_price: float, target_price: float) -> bool:
        """가격 차이 허용 범위 체크"""
        try:
            # 가격 유효성 검사
            source_price = float(source_price or 0)
            target_price = float(target_price or 0)
            
            # 가격 범위 체크
            if not (self.price_min <= source_price <= self.price_max and 
                    self.price_min <= target_price <= self.price_max):
                return False
                
            # 가격이 0 이하인 경우 호환
            if source_price <= 0 or target_price <= 0:
                return True
                
            # 가격 차이 계산
            price_diff = abs(source_price - target_price) / max(source_price, target_price)
            return price_diff <= self.price_diff_threshold
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error in price compatibility check: {e}")
            # 가격 오류 시 호환되지 않음
            return False

    def find_matches(
        self,
        source_product: Product,
        candidate_products: List[Product],
        min_text_similarity: float = None,
        min_image_similarity: float = None,
        min_combined_similarity: float = None,
        max_matches: int = None,
    ) -> List[MatchResult]:
        """Find best matches among candidates according to multimodal similarity.
        
        Args:
            source_product: 매칭 원본 제품
            candidate_products: 매칭 후보 제품 목록
            min_text_similarity: 최소 텍스트 유사도
            min_image_similarity: 최소 이미지 유사도
            min_combined_similarity: 최소 종합 유사도
            max_matches: 최대 반환 매칭 수
            
        Returns:
            유사도 내림차순으로 정렬된 매칭 결과 목록
        """
        try:
            # 기본값 설정
            if min_text_similarity is None:
                min_text_similarity = self.text_similarity_threshold
            if min_image_similarity is None:
                min_image_similarity = self.image_similarity_threshold
            if min_combined_similarity is None:
                min_combined_similarity = self.similarity_threshold
            if max_matches is None:
                max_matches = self.batch_size

            # 입력 검증
            if not source_product:
                self.logger.warning("Invalid source product")
                return []
                
            if not candidate_products:
                self.logger.debug(f"No candidate products for {source_product.name}")
                return []

            self.logger.debug(
                f"Finding matches for {source_product.name} among {len(candidate_products)} candidates"
            )

            # 매칭 작업 구성
            candidate_tasks = []
            for candidate in candidate_products:
                # 자기 자신과 매칭 방지
                if candidate.id == source_product.id:
                    continue
                    
                # 필터링 키워드가 포함된 제품 제외 (옵션)
                if self.matching_settings.get("filter_keywords_enabled", False):
                    if self._contains_filter_keywords(candidate.name):
                        continue
                
                candidate_tasks.append(candidate)
            
            # 후보가 없다면 빈 목록 반환
            if not candidate_tasks:
                self.logger.debug(f"No valid candidates for {source_product.name} after filtering")
                return []

            # 병렬 처리를 위한 작업자 수 설정 (최대 작업자 수와 후보 수 중 작은 값)
            num_workers = min(self.max_workers, len(candidate_tasks))
            
            # 매칭 작업 병렬 처리
            matches = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_candidate = {
                    executor.submit(
                        self._process_candidate_match,
                        source_product,
                        candidate,
                        min_text_similarity,
                        min_image_similarity,
                    ): candidate
                    for candidate in candidate_tasks
                }

                for future in concurrent.futures.as_completed(future_to_candidate):
                    candidate = future_to_candidate[future]
                    try:
                        match_result = future.result(timeout=self.extraction_timeout)
                        if match_result and match_result.combined_similarity >= min_combined_similarity:
                            matches.append(match_result)
                    except concurrent.futures.TimeoutError:
                        self.logger.warning(f"Matching timeout for candidate {candidate.id}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error matching candidate {candidate.id}: {str(e)}", exc_info=True)
                        continue

                    # 정중한 지연 적용 (ms -> s)
                    if self.politeness_delay > 0:
                        time.sleep(self.politeness_delay / 1000.0)

            # 매칭 결과가 없다면 빈 목록 반환
            if not matches:
                self.logger.debug(f"No matches found for {source_product.name}")
                return []
                
            # 유사도 기준 내림차순 정렬
            matches.sort(key=lambda m: m.combined_similarity, reverse=True)

            # 상위 N개 매칭 결과 반환
            return matches[:max_matches]
            
        except Exception as e:
            self.logger.error(f"Error in find_matches for {source_product.name if source_product else 'None'}: {str(e)}", exc_info=True)
            return []

    def _process_candidate_match(
        self,
        source_product: Product,
        candidate: Product,
        min_text_similarity: float,
        min_image_similarity: float,
    ) -> Optional[MatchResult]:
        """Process a single candidate match with retries"""
        retries = 0
        while retries < self.max_retries:
            try:
                # 필터링: 카테고리 호환성 검사
                if not self._check_category_compatibility(source_product.category, candidate.category):
                    return None
                    
                # 필터링: 가격 호환성 검사
                if not self._check_price_compatibility(source_product.price, candidate.price):
                    return None
                
                # 제품명 정제 (설정된 경우)
                source_name = self._clean_product_name(source_product.name) if self.auto_clean_product_names else source_product.name
                candidate_name = self._clean_product_name(candidate.name) if self.auto_clean_product_names else candidate.name
                
                # 텍스트 유사도 계산
                text_similarity = self.text_matcher.calculate_similarity(
                    source_name, candidate_name
                )

                # 텍스트 유사도 최소 임계값 확인
                if text_similarity < min_text_similarity:
                    return None

                # 이미지 유사도 계산 (이미지가 있는 경우)
                image_similarity = 0.0
                has_both_images = False

                if source_product.image_url and candidate.image_url:
                    has_both_images = True
                    image_similarity = self.image_matcher.calculate_similarity(
                        source_product.image_url,
                        candidate.image_url
                    )

                    # 이미지 유사도 최소 임계값 확인
                    if image_similarity < min_image_similarity:
                        return None

                # 적응형 가중치 계산
                if self.use_adaptive_weights:
                    text_weight, image_weight = self._calculate_adaptive_weights(
                        text_similarity, image_similarity, has_both_images
                    )
                else:
                    text_weight, image_weight = self.text_weight, self.image_weight

                # 종합 유사도 계산
                combined_similarity = (text_similarity * text_weight) + (image_similarity * image_weight)

                # 가격 차이 계산
                price_difference = 0.0
                price_difference_percent = 0.0
                
                if source_product.price and candidate.price:
                    price_difference = candidate.price - source_product.price
                    if source_product.price > 0:
                        price_difference_percent = (price_difference / source_product.price) * 100

                # 매칭 결과 생성
                return MatchResult(
                    source_product=source_product,
                    matched_product=candidate,
                    text_similarity=text_similarity,
                    image_similarity=image_similarity,
                    combined_similarity=combined_similarity,
                    text_weight=text_weight,
                    image_weight=image_weight,
                    has_both_images=has_both_images,
                    price_difference=price_difference,
                    price_difference_percent=price_difference_percent
                )

            except Exception as e:
                retries += 1
                self.logger.warning(f"Retry {retries}/{self.max_retries} for matching {candidate.id}: {str(e)}")
                
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries reached for candidate {candidate.id}: {str(e)}")
                    return None
                
                # 재시도 지연 시간 계산
                if self.exponential_backoff:
                    wait_time = (2 ** retries) * 0.1  # exponential backoff
                else:
                    wait_time = 0.1  # constant delay
                
                time.sleep(wait_time)
                continue

    def _clean_product_name(self, name: str) -> str:
        """Clean product name if auto_clean_product_names is enabled"""
        if not name:
            return name
            
        try:
            # Remove common noise patterns
            name = re.sub(r'\[.*?\]', '', name)  # Remove content in square brackets
            name = re.sub(r'\(.*?\)', '', name)  # Remove content in parentheses
            name = re.sub(r'[^\w\s가-힣一-龥-]', ' ', name)  # Replace special chars with space (keep Korean/Chinese chars)
            name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with single space
            name = name.strip()
            
            return name
        except Exception as e:
            self.logger.warning(f"Error cleaning product name: {e}")
            return name  # 오류 시 원본 반환

    def find_best_match(
        self,
        source_product: Product,
        candidate_products: List[Product],
        min_text_similarity: float = None,
        min_image_similarity: float = None,
        min_combined_similarity: float = None,
    ) -> Optional[MatchResult]:
        """Find the best match for a product among candidates.

        네이버 쇼핑 매칭 기준:
        1. 상품 이름으로 검색하여 동일 상품 찾기
        2. 이미지로 제품 비교 (이미지 비교가 어려운 경우 규격 확인)
        3. 동일 상품으로 판단되면 가장 낮은 가격의 상품 선택
        
        Args:
            source_product: 매칭 원본 제품
            candidate_products: 매칭 후보 제품 목록
            min_text_similarity: 최소 텍스트 유사도
            min_image_similarity: 최소 이미지 유사도
            min_combined_similarity: 최소 종합 유사도
            
        Returns:
            가장 적합한 매칭 결과 또는 None
        """
        try:
            # 기본값 설정
            if min_text_similarity is None:
                min_text_similarity = self.text_similarity_threshold
            if min_image_similarity is None:
                min_image_similarity = self.image_similarity_threshold
            if min_combined_similarity is None:
                min_combined_similarity = self.similarity_threshold

            # 매칭된 모든 상품 가져오기
            matches = self.find_matches(
                source_product,
                candidate_products,
                min_text_similarity=min_text_similarity,
                min_image_similarity=min_image_similarity,
                min_combined_similarity=min_combined_similarity,
                max_matches=self.batch_size
            )

            # 매칭 결과가 있다면, 매칭된 상품들 중 가장 적합한 상품 선택
            if matches:
                # 확실히 매칭된 상품들만 필터링 (combined_similarity > similarity_threshold)
                confident_matches = [m for m in matches if m.combined_similarity > self.similarity_threshold]

                if confident_matches:
                    # 확실한 매칭 중 가격순 정렬 (가격이 가장 낮은 것)
                    confident_matches.sort(key=lambda m: m.matched_product.price if m.matched_product.price else float('inf'))
                    best_match = confident_matches[0]
                    self.logger.info(f"Found best confident match for {source_product.name}: {best_match.matched_product.name} (similarity: {best_match.combined_similarity:.3f}, price: {best_match.matched_product.price})")
                    return best_match
                else:
                    # 낮은 신뢰도 매칭 중 가장 유사도 높은 것
                    best_match = matches[0]
                    self.logger.info(f"Found best match (low confidence) for {source_product.name}: {best_match.matched_product.name} (similarity: {best_match.combined_similarity:.3f})")
                    return best_match
            else:
                self.logger.info(f"No matches found for {source_product.name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in find_best_match for {source_product.name if source_product else 'None'}: {str(e)}", exc_info=True)
            return None

    def calculate_price_difference(
        self, source_price: float, matched_price: float
    ) -> Tuple[float, float]:
        """
        Calculate absolute and percentage price differences.

        Args:
            source_price: Source product price
            matched_price: Matched product price

        Returns:
            Tuple of (price_difference, price_difference_percent)
        """
        try:
            # 가격 유효성 검사
            source_price = float(source_price or 0)
            matched_price = float(matched_price or 0)
            
            if source_price <= 0:
                return 0.0, 0.0

            price_difference = matched_price - source_price
            price_difference_percent = (price_difference / source_price) * 100

            return price_difference, price_difference_percent
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"Error calculating price difference: {e}")
            return 0.0, 0.0

    def calculate_similarity(
        self, product1, product2, max_dimension: int = None
    ) -> float:
        """
        Calculate the combined similarity between two products or combine pre-calculated similarities.

        Args:
            product1: First product or text similarity score
            product2: Second product or image similarity score
            max_dimension: 이미지 처리를 위한 최대 해상도 (속도 최적화)

        Returns:
            Combined similarity score between 0.0 and 1.0
        """
        try:
            # Check if inputs are already similarity scores
            if isinstance(product1, (float, int)) and isinstance(product2, (float, int)):
                # Inputs are similarity scores, combine them directly
                text_sim = float(product1)
                image_sim = float(product2)

                # 적응형 가중치 계산
                if self.use_adaptive_weights:
                    text_weight, image_weight = self._calculate_adaptive_weights(
                        text_sim, image_sim, True
                    )
                else:
                    text_weight = self.text_weight
                    image_weight = self.image_weight

            elif isinstance(product1, Product) and isinstance(product2, Product):
                # Calculate text similarity
                text_sim = self.text_matcher.calculate_similarity(
                    product1.name, product2.name
                )

                # Calculate image similarity (if available)
                image_sim = 0.0
                has_images = False

                if product1.image_url and product2.image_url:
                    has_images = True
                    image_sim = self.image_matcher.calculate_similarity(
                        product1.image_url, product2.image_url, max_dimension=max_dimension
                    )

                # 적응형 가중치 계산
                if self.use_adaptive_weights:
                    text_weight, image_weight = self._calculate_adaptive_weights(
                        text_sim, image_sim, has_images
                    )
                else:
                    text_weight = self.text_weight
                    image_weight = self.image_weight
            else:
                self.logger.error(
                    f"Invalid argument types for calculate_similarity: {type(product1)}, {type(product2)}"
                )
                return 0.0

            # Combine similarities with weights
            combined_sim = text_weight * text_sim + image_weight * image_sim

            return float(combined_sim)
            
        except Exception as e:
            self.logger.error(f"Error in calculate_similarity: {str(e)}", exc_info=True)
            return 0.0

    def _contains_filter_keywords(self, text: str) -> bool:
        """특정 키워드가 포함되어 있는지 확인"""
        if not text:
            return False

        try:
            text = text.lower()
            for keyword in self.filter_keywords:
                if keyword in text:
                    return True
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking filter keywords: {e}")
            return False  # 오류 발생 시 필터링하지 않음

    def batch_find_matches(
        self,
        query_products: List[Product],
        candidate_products: List[Product],
        max_results_per_query: int = None,
        min_similarity: float = None,
    ) -> Dict[str, List[Tuple[Product, float]]]:
        """
        여러 제품에 대해 일괄적으로 매칭 검색 수행

        Args:
            query_products: 검색할 제품 목록
            candidate_products: 후보 제품 목록
            max_results_per_query: 제품당 최대 결과 수
            min_similarity: 최소 유사도 (None이면 기본값 사용)

        Returns:
            제품 ID를 키로 하고 (매칭된 제품, 유사도) 튜플 리스트를 값으로 하는 딕셔너리
        """
        try:
            # 기본값 설정
            if min_similarity is None:
                min_similarity = self.similarity_threshold
            if max_results_per_query is None:
                max_results_per_query = self.batch_size

            # 입력 검증
            if not query_products or not candidate_products:
                self.logger.warning("Invalid input for batch_find_matches")
                return {}
                
            self.logger.info(f"Running batch matching for {len(query_products)} products against {len(candidate_products)} candidates")

            results = {}
            processed_count = 0
            
            # 각 쿼리 제품에 대해 매칭 검색
            for query_product in query_products:
                if not query_product or not query_product.id:
                    continue
                    
                # 진행 상황 로깅 (10개 단위로)
                processed_count += 1
                if processed_count % 10 == 0:
                    self.logger.info(f"Batch matching progress: {processed_count}/{len(query_products)}")
                
                # 매칭 검색
                matches = self.find_matches(
                    query_product,
                    candidate_products,
                    min_combined_similarity=min_similarity,
                    max_matches=max_results_per_query,
                )

                # 매칭 결과를 (제품, 유사도) 튜플 형태로 변환
                product_matches = [
                    (match.matched_product, match.combined_similarity) for match in matches
                ]

                # 결과 저장
                results[query_product.id] = product_matches

            self.logger.info(f"Batch matching completed: {len(results)} products matched")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch_find_matches: {str(e)}", exc_info=True)
            return {}
