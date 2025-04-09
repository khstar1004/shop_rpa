import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .text_matcher import TextMatcher
from .image_matcher import ImageMatcher
from ..data_models import Product, MatchResult
from utils.caching import FileCache


class MultiModalMatcher:
    """
    Multimodal product matcher that combines text and image similarity.
    
    업데이트된 기능:
    - 매뉴얼 요구사항에 맞게 매칭 로직 강화
    - 이미지와 규격이 동일한 경우만 매칭으로 판단
    - 가격차이 계산 및 필터링 로직 추가
    - 적응형 가중치 조정 시스템 추가
    - 카테고리 기반 필터링 추가
    - 유사도 점수 향상을 위한 보너스 시스템 구현
    """
    
    def __init__(
        self,
        text_matcher: TextMatcher,
        image_matcher: ImageMatcher,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
        similarity_threshold: float = 0.7,
        min_price_diff_percent: float = 10.0,
        use_adaptive_weights: bool = True
    ):
        self.text_matcher = text_matcher
        self.image_matcher = image_matcher
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.similarity_threshold = similarity_threshold
        self.min_price_diff_percent = min_price_diff_percent
        
        # 적응형 가중치 사용 여부
        self.use_adaptive_weights = use_adaptive_weights
        
        # 최소 가중치 설정으로 한쪽에 너무 치우치지 않도록 방지
        self.min_text_weight = 0.3
        self.min_image_weight = 0.2
        
        # For ignoring small price differences (less than min_price_diff_percent)
        self.price_diff_threshold = min_price_diff_percent / 100.0
        
        # 카테고리 매핑 정의 (카테고리 호환성 매트릭스)
        self.category_compatibility = {
            # 주요 카테고리: 완전 호환 카테고리 목록
            '전자제품': ['가전', '디지털', '컴퓨터', 'IT기기', '휴대폰', '카메라'],
            '패션': ['의류', '신발', '가방', '액세서리', '쥬얼리'],
            '식품': ['식료품', '음료', '건강식품', '간식'],
            '뷰티': ['화장품', '헤어', '바디', '스킨케어'],
            '생활용품': ['주방', '욕실', '청소', '수납'],
            # 추가 카테고리도 필요에 따라 정의
        }
        
        # 카테고리 별칭 (다른 표현으로 된 동일 카테고리)
        self.category_aliases = {
            '디지털': ['전자', '가전제품', '전자기기'],
            '의류': ['옷', '패션의류', '의상'],
            '식료품': ['음식', '먹거리', '식품'],
            '화장품': ['메이크업', '코스메틱', '미용'],
            # 추가 별칭 필요시 정의
        }
        
        self.logger = logging.getLogger(__name__)

    def find_matches(
        self, 
        source_product: Product, 
        candidate_products: List[Product],
        min_text_similarity: float = 0.35,  # 더 낮은 텍스트 임계값으로 매칭 기회 증가
        min_image_similarity: float = 0.0,  # 이미지 없어도 매칭 가능하도록 설정
        min_combined_similarity: float = 0.5,  # 결합 유사도 임계값도 낮춤
        max_matches: int = 10  # 더 많은 매칭 후보 반환
    ) -> List[MatchResult]:
        """Find best matches among candidates according to multimodal similarity."""
        
        if not candidate_products:
            self.logger.debug(f"No candidate products for {source_product.name}")
            return []
            
        self.logger.debug(f"Finding matches for {source_product.name} among {len(candidate_products)} candidates")
        
        matches = []
        for candidate in candidate_products:
            # Skip if candidate is the same as source
            if source_product.id == candidate.id:
                continue
                
            # 카테고리 호환성 체크
            category_compatible = self._check_category_compatibility(
                getattr(source_product, 'category', None),
                getattr(candidate, 'category', None)
            )
            
            # 호환되지 않는 카테고리면 스킵 (카테고리가 없으면 통과)
            if not category_compatible and hasattr(source_product, 'category') and source_product.category:
                self.logger.debug(f"Skipping product {candidate.name} due to incompatible category")
                continue
                
            # Calculate text similarity
            text_similarity = self.text_matcher.calculate_similarity(
                source_product.name, 
                candidate.name
            )
            
            # 제품 규격 추출 및 비교
            if hasattr(self.text_matcher, 'extract_product_specs'):
                source_specs = self.text_matcher.extract_product_specs(source_product.name)
                candidate_specs = self.text_matcher.extract_product_specs(candidate.name)
                
                # 규격 일치 여부 확인 (같은 크기/용량 등)
                spec_match = False
                for spec_type, source_values in source_specs.items():
                    if spec_type in candidate_specs:
                        for s_val in source_values:
                            for c_val in candidate_specs[spec_type]:
                                if self.text_matcher._normalize_spec(s_val) == self.text_matcher._normalize_spec(c_val):
                                    spec_match = True
                                    break
                            if spec_match:
                                break
                    if spec_match:
                        break
                        
                # 규격 일치시 텍스트 유사도 가중
                if spec_match:
                    text_similarity = min(1.0, text_similarity * 1.1)
            
            # Calculate image similarity if both have image URLs
            image_similarity = 0.0
            has_both_images = False
            
            if source_product.image_url and candidate.image_url:
                has_both_images = True
                image_similarity = self.image_matcher.calculate_similarity(
                    source_product.image_url,
                    candidate.image_url
                )
            
            # 두 제품 모두 이미지가 없는 경우 특별 처리
            if not has_both_images:
                # 텍스트 유사도가 매우 높으면(0.8 이상) 이미지 없어도 높은 점수 부여
                if text_similarity >= 0.8:
                    image_similarity = 0.7  # 텍스트가 매우 유사하면 이미지도 유사할 가능성 높음
                # 텍스트 유사도가 적절히 높으면(0.6 이상) 중간 점수 부여
                elif text_similarity >= 0.6:
                    image_similarity = 0.5
                # 텍스트 유사도가 최소 기준 이상이면 낮은 점수 부여
                elif text_similarity >= min_text_similarity:
                    image_similarity = 0.3
            
            # Skip if below individual thresholds
            # 텍스트 유사도는 필수지만, 이미지 유사도는 없어도 됨
            if text_similarity < min_text_similarity:
                continue
            
            # 적응형 가중치 계산
            if self.use_adaptive_weights:
                # 텍스트/이미지 품질에 따라 가중치 동적 조정
                text_weight, image_weight = self._calculate_adaptive_weights(
                    text_similarity, 
                    image_similarity,
                    has_both_images
                )
            else:
                # 고정 가중치 사용
                text_weight = self.text_weight
                image_weight = self.image_weight
            
            # Calculate combined similarity
            combined_similarity = (
                text_weight * text_similarity + 
                image_weight * image_similarity
            )
            
            # Skip if below combined threshold
            if combined_similarity < min_combined_similarity:
                continue
                
            # Always include products with very high text similarity regardless of image
            if text_similarity >= 0.9:
                self.logger.info(f"High text similarity match found: {candidate.name} (Text: {text_similarity:.2f})")
            
            # Check price (if reference_price is provided)
            price_filter_passed = True
            price_difference = 0.0
            price_difference_percent = 0.0
            
            if hasattr(source_product, 'price') and source_product.price > 0 and candidate.price > 0:
                # Calculate price difference
                price_difference = candidate.price - source_product.price
                price_difference_percent = price_difference / source_product.price * 100
                
                # Apply 10% rule if enabled
                if self.price_diff_threshold > 0:
                    # Allow lower prices or prices with significant difference
                    price_filter_passed = (
                        price_difference_percent <= 0 or 
                        abs(price_difference_percent) >= self.min_price_diff_percent
                    )
            
            if not price_filter_passed:
                # 가격 차이가 작아서 필터링됨
                self.logger.debug(f"Skipping product {candidate.name} due to insufficient price difference ({price_difference_percent:.1f}%)")
                continue
            
            # Create match result with additional metadata
            match = MatchResult(
                source_product=source_product,
                matched_product=candidate,
                text_similarity=text_similarity,
                image_similarity=image_similarity,
                combined_similarity=combined_similarity,
                price_difference=price_difference,
                price_difference_percent=price_difference_percent,
                # 추가 메타데이터
                text_weight=text_weight,
                image_weight=image_weight,
                category_match=category_compatible,
                has_both_images=has_both_images
            )
            
            matches.append(match)
        
        # Sort by combined similarity (descending)
        matches.sort(key=lambda m: m.combined_similarity, reverse=True)
        
        # Return top matches
        return matches[:max_matches]
        
    def find_best_match(
        self, 
        source_product: Product, 
        candidate_products: List[Product],
        min_text_similarity: float = 0.4,
        min_image_similarity: float = 0.3,
        min_combined_similarity: float = 0.6
    ) -> Optional[MatchResult]:
        """Find the best match for a product among candidates.
        
        네이버 쇼핑 매칭 기준:
        1. 상품 이름으로 검색하여 동일 상품 찾기
        2. 이미지로 제품 비교 (이미지 비교가 어려운 경우 규격 확인)
        3. 동일 상품으로 판단되면 가장 낮은 가격의 상품 선택
        """
        # 매칭된 모든 상품 가져오기
        matches = self.find_matches(
            source_product,
            candidate_products,
            min_text_similarity,
            min_image_similarity,
            min_combined_similarity
        )
        
        # 매칭 결과가 있다면, 매칭된 상품들 중 가장 가격이 낮은 상품 선택
        if matches:
            # 확실히 매칭된 상품들만 필터링 (combined_similarity > 0.7)
            confident_matches = [m for m in matches if m.combined_similarity > 0.7]
            
            if confident_matches:
                # 확실한 매칭 중 가격순 정렬
                confident_matches.sort(key=lambda m: m.matched_product.price)
                return confident_matches[0]
            else:
                # 낮은 신뢰도 매칭 중 가장 유사도 높은 것
                return matches[0]
        else:
            return None
    
    def calculate_price_difference(self, source_price: float, matched_price: float) -> Tuple[float, float]:
        """
        Calculate absolute and percentage price differences.
        
        Args:
            source_price: Source product price
            matched_price: Matched product price
            
        Returns:
            Tuple of (price_difference, price_difference_percent)
        """
        if source_price <= 0:
            return 0, 0
            
        price_difference = matched_price - source_price
        price_difference_percent = (price_difference / source_price) * 100
        
        return price_difference, price_difference_percent

    def calculate_similarity(self, product1, product2) -> float:
        """
        Calculate the combined similarity between two products or combine pre-calculated similarities.
        
        Args:
            product1: First product or text similarity score
            product2: Second product or image similarity score
            
        Returns:
            Combined similarity score between 0.0 and 1.0
        """
        # Check if inputs are already similarity scores
        if isinstance(product1, (float, int)) and isinstance(product2, (float, int)):
            # Inputs are similarity scores, combine them directly
            text_sim = float(product1)
            image_sim = float(product2)
            
            # 적응형 가중치 계산
            if self.use_adaptive_weights:
                text_weight, image_weight = self._calculate_adaptive_weights(text_sim, image_sim, True)
            else:
                text_weight = self.text_weight
                image_weight = self.image_weight
                
        elif isinstance(product1, Product) and isinstance(product2, Product):
            # Calculate text similarity
            text_sim = self.text_matcher.calculate_similarity(
                product1.name, 
                product2.name
            )
            
            # Calculate image similarity (if available)
            image_sim = 0.0
            has_images = False
            
            if product1.image_url and product2.image_url:
                has_images = True
                image_sim = self.image_matcher.calculate_similarity(
                    product1.image_url, 
                    product2.image_url
                )
                
            # 적응형 가중치 계산
            if self.use_adaptive_weights:
                text_weight, image_weight = self._calculate_adaptive_weights(text_sim, image_sim, has_images)
            else:
                text_weight = self.text_weight
                image_weight = self.image_weight
        else:
            self.logger.error(f"Invalid argument types for calculate_similarity: {type(product1)}, {type(product2)}")
            return 0.0
        
        # Combine similarities with weights
        combined_sim = (
            text_weight * text_sim + 
            image_weight * image_sim
        )
        
        return float(combined_sim)
        
    def _calculate_adaptive_weights(
        self, 
        text_similarity: float,
        image_similarity: float,
        has_images: bool
    ) -> Tuple[float, float]:
        """텍스트와 이미지 유사도 신뢰도에 기반한 가중치 동적 계산"""
        if not has_images:
            # 이미지가 없으면 텍스트에 높은 가중치
            return 0.9, 0.1
            
        # 텍스트 유사도가 높은 경우 (0.8 이상)
        if text_similarity >= 0.8:
            # 이미지 유사도도 높으면 (0.7 이상) 균형있게
            if image_similarity >= 0.7:
                text_weight = 0.6
                image_weight = 0.4
            # 이미지 유사도가 낮으면 텍스트에 가중치
            else:
                text_weight = 0.8
                image_weight = 0.2
        # 텍스트 유사도가 중간인 경우 (0.5~0.8)
        elif text_similarity >= 0.5:
            # 이미지 유사도가 높으면 (0.7 이상) 이미지에 가중치
            if image_similarity >= 0.7:
                text_weight = 0.4
                image_weight = 0.6
            # 이미지 유사도가 중간이면 균형있게
            elif image_similarity >= 0.5:
                text_weight = 0.5
                image_weight = 0.5
            # 이미지 유사도가 낮으면 텍스트에 가중치
            else:
                text_weight = 0.7
                image_weight = 0.3
        # 텍스트 유사도가 낮은 경우 (0.5 미만)
        else:
            # 이미지 유사도가 높으면 이미지에 무게
            if image_similarity >= 0.7:
                text_weight = 0.3
                image_weight = 0.7
            # 그 외의 경우 기본 가중치
            else:
                text_weight = self.text_weight
                image_weight = self.image_weight
        
        # 최소 가중치는 보장
        text_weight = max(text_weight, self.min_text_weight)
        image_weight = max(image_weight, self.min_image_weight)
        
        # 가중치 합이 1이 되도록 정규화
        total = text_weight + image_weight
        text_weight = text_weight / total
        image_weight = image_weight / total
        
        return text_weight, image_weight
        
    def _check_category_compatibility(
        self, 
        source_category: Optional[str], 
        candidate_category: Optional[str]
    ) -> bool:
        """두 카테고리의 호환성 체크"""
        # 카테고리가 없으면 호환된다고 간주
        if not source_category or not candidate_category:
            return True
            
        # 카테고리 정규화
        source_category = source_category.lower().strip()
        candidate_category = candidate_category.lower().strip()
        
        # 정확히 일치하면 호환
        if source_category == candidate_category:
            return True
            
        # 카테고리 별칭 확인
        normalized_source = self._normalize_category(source_category)
        normalized_candidate = self._normalize_category(candidate_category)
        
        if normalized_source == normalized_candidate:
            return True
            
        # 카테고리 호환성 매트릭스 확인
        for main_category, compatible_categories in self.category_compatibility.items():
            if (normalized_source == main_category.lower() and 
                normalized_candidate in [c.lower() for c in compatible_categories]):
                return True
            if (normalized_candidate == main_category.lower() and 
                normalized_source in [c.lower() for c in compatible_categories]):
                return True
                
        # 위의 조건에 해당되지 않으면 호환되지 않음
        return False
        
    def _normalize_category(self, category: str) -> str:
        """카테고리명 정규화 (별칭 처리)"""
        category = category.lower().strip()
        
        # 별칭 확인
        for main_category, aliases in self.category_aliases.items():
            if category in [a.lower() for a in aliases]:
                return main_category.lower()
                
        # 별칭이 없으면 원래 카테고리 반환
        return category
        
    def batch_find_matches(
        self,
        query_products: List[Product],
        candidate_products: List[Product],
        max_results_per_query: int = 5,
        min_similarity: float = None
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
        if min_similarity is None:
            min_similarity = self.similarity_threshold
            
        results = {}
        
        for query_product in query_products:
            matches = self.find_matches(
                query_product,
                candidate_products,
                min_combined_similarity=min_similarity,
                max_matches=max_results_per_query
            )
            
            # 매칭 결과를 (제품, 유사도) 튜플 형태로 변환
            product_matches = [
                (match.matched_product, match.combined_similarity)
                for match in matches
            ]
            
            # 결과 저장
            results[query_product.id] = product_matches
            
        return results 