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
    """
    
    def __init__(
        self,
        text_matcher: TextMatcher,
        image_matcher: ImageMatcher,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
        similarity_threshold: float = 0.7,
        min_price_diff_percent: float = 10.0
    ):
        self.text_matcher = text_matcher
        self.image_matcher = image_matcher
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.similarity_threshold = similarity_threshold
        self.min_price_diff_percent = min_price_diff_percent
        
        # For ignoring small price differences (less than min_price_diff_percent)
        self.price_diff_threshold = min_price_diff_percent / 100.0
        
        self.logger = logging.getLogger(__name__)

    def find_matches(
        self, 
        source_product: Product, 
        candidate_products: List[Product],
        min_text_similarity: float = 0.4,
        min_image_similarity: float = 0.3,
        min_combined_similarity: float = 0.6,
        max_matches: int = 5
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
                
            # Calculate text similarity
            text_similarity = self.text_matcher.calculate_similarity(
                source_product.name, 
                candidate.name
            )
            
            # Calculate image similarity if both have image URLs
            image_similarity = 0.0
            if source_product.image_url and candidate.image_url:
                image_similarity = self.image_matcher.calculate_similarity(
                    source_product.image_url,
                    candidate.image_url
                )
            
            # Skip if below individual thresholds
            if text_similarity < min_text_similarity or image_similarity < min_image_similarity:
                continue
                
            # Calculate combined similarity
            combined_similarity = (
                self.text_weight * text_similarity + 
                self.image_weight * image_similarity
            )
            
            # Skip if below combined threshold
            if combined_similarity < min_combined_similarity:
                continue
                
            # Calculate price difference - 매뉴얼 요구사항
            if source_product.price > 0 and candidate.price > 0:
                price_difference = candidate.price - source_product.price
                price_difference_percent = (price_difference / source_product.price) * 100
            else:
                price_difference = 0
                price_difference_percent = 0
                
            # 매뉴얼 요구사항: 네이버 쇼핑 기본수량 없는 경우 처리
            # If no quantity info and price difference is too small, skip
            if (getattr(candidate, 'min_order_quantity', None) is None or 
                candidate.min_order_quantity <= 0) and \
                0 < price_difference_percent < self.min_price_diff_percent:
                continue
                
            # Create match result
            match = MatchResult(
                source_product=source_product,
                matched_product=candidate,
                text_similarity=text_similarity,
                image_similarity=image_similarity,
                combined_similarity=combined_similarity,
                price_difference=price_difference,
                price_difference_percent=price_difference_percent
            )
            matches.append(match)
        
        # Sort by combined similarity (highest first)
        matches.sort(key=lambda x: x.combined_similarity, reverse=True)
        
        # 매뉴얼 요구사항: 이미지와 규격이 동일한 경우만 마지막 확인
        verified_matches = []
        for match in matches[:max_matches]:
            # 이미지 유사도가 기준 이상인 경우만 채택
            if match.image_similarity >= 0.7:
                verified_matches.append(match)
            else:
                self.logger.debug(f"Rejecting match due to low image similarity: {match.matched_product.name}")
        
        # 매칭된 상품이 없으면 동일상품 없음으로 처리
        if not verified_matches and matches:
            self.logger.debug(f"No matches with sufficient image similarity for {source_product.name}")
            
        return verified_matches[:max_matches]
        
    def find_best_match(
        self, 
        source_product: Product, 
        candidate_products: List[Product],
        min_text_similarity: float = 0.4,
        min_image_similarity: float = 0.3,
        min_combined_similarity: float = 0.6
    ) -> Optional[MatchResult]:
        """Find the best match for a product among candidates."""
        matches = self.find_matches(
            source_product,
            candidate_products,
            min_text_similarity,
            min_image_similarity,
            min_combined_similarity,
            max_matches=1
        )
        
        if matches:
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

    def calculate_similarity(self, product1: Product, product2: Product) -> float:
        """
        Calculate the combined similarity between two products.
        
        Args:
            product1: First product
            product2: Second product
            
        Returns:
            Combined similarity score between 0.0 and 1.0
        """
        # Calculate text similarity
        text_sim = self.text_matcher.calculate_similarity(
            product1.name, 
            product2.name
        )
        
        # Calculate image similarity (if available)
        image_sim = 0.0
        if product1.image_url and product2.image_url:
            image_sim = self.image_matcher.calculate_similarity(
                product1.image_url, 
                product2.image_url
            )
        
        # Combine similarities with weights
        combined_sim = (
            self.text_weight * text_sim + 
            self.image_weight * image_sim
        )
        
        return float(combined_sim)
    
    def batch_find_matches(
        self,
        query_products: List[Product],
        candidate_products: List[Product],
        max_results_per_query: int = 5,
        min_similarity: float = None
    ) -> Dict[str, List[Tuple[Product, float]]]:
        """
        Find matches for multiple query products in batch.
        
        Args:
            query_products: List of products to find matches for
            candidate_products: List of products to search through
            max_results_per_query: Maximum results per query product
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary mapping product IDs to their matches
        """
        results = {}
        
        for query_product in query_products:
            matches = self.find_matches(
                query_product=query_product,
                candidate_products=candidate_products,
                max_results=max_results_per_query,
                min_similarity=min_similarity
            )
            results[query_product.id] = matches
            
        return results 