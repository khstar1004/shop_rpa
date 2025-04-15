"""
Base matcher interface for all matching implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import logging

from ..data_models import Product, MatchResult


class BaseMatcher(ABC):
    """Base class for all matchers."""

    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """Initialize the base matcher.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def match(self, source_product: Product, target_product: Product) -> float:
        """Calculate similarity score between source and target products.
        
        Args:
            source_product: Source product to match from
            target_product: Target product to match against
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def find_matches(
        self,
        source_product: Product,
        candidate_products: List[Product],
        min_text_similarity: float = None,
        min_image_similarity: float = None,
        min_combined_similarity: float = None,
        max_matches: int = None,
    ) -> List[MatchResult]:
        """Find best matches among candidates according to similarity.
        
        Args:
            source_product: Source product to match from
            candidate_products: List of candidate products to match against
            min_text_similarity: Minimum text similarity threshold
            min_image_similarity: Minimum image similarity threshold
            min_combined_similarity: Minimum combined similarity threshold
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match results sorted by similarity (descending)
        """
        pass

    @abstractmethod
    def find_best_match(
        self,
        source_product: Product,
        candidate_products: List[Product],
        min_text_similarity: float = None,
        min_image_similarity: float = None,
        min_combined_similarity: float = None,
    ) -> Optional[MatchResult]:
        """Find the best match for a product among candidates.
        
        Args:
            source_product: Source product to match from
            candidate_products: List of candidate products to match against
            min_text_similarity: Minimum text similarity threshold
            min_image_similarity: Minimum image similarity threshold
            min_combined_similarity: Minimum combined similarity threshold
            
        Returns:
            Best match result or None if no match found
        """
        pass

    @abstractmethod
    def batch_find_matches(
        self,
        query_products: List[Product],
        candidate_products: List[Product],
        max_results_per_query: int = None,
        min_similarity: float = None,
    ) -> Dict[str, List[Tuple[Product, float]]]:
        """Batch find matches for multiple products.
        
        Args:
            query_products: List of products to match from
            candidate_products: List of candidate products to match against
            max_results_per_query: Maximum number of matches per query product
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary mapping query product IDs to list of (matched_product, similarity) tuples
        """
        pass 