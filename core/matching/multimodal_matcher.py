import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .text_matcher import TextMatcher
from .image_matcher import ImageMatcher
from ..data_models import Product
from utils.caching import FileCache


class MultiModalMatcher:
    """
    Combines text and image matching to provide multimodal similarity calculation
    between products.
    """
    
    def __init__(
        self, 
        text_weight: float = 0.6, 
        image_weight: float = 0.4,
        text_matcher: Optional[TextMatcher] = None,
        image_matcher: Optional[ImageMatcher] = None,
        cache: Optional[FileCache] = None,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize the MultiModalMatcher.
        
        Args:
            text_weight: Weight given to text similarity (default: 0.6)
            image_weight: Weight given to image similarity (default: 0.4)
            text_matcher: Existing TextMatcher instance, or None to create new
            image_matcher: Existing ImageMatcher instance, or None to create new
            cache: Optional cache for results
            similarity_threshold: Threshold to consider two products as similar
        """
        self.logger = logging.getLogger(__name__)
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.similarity_threshold = similarity_threshold
        self.cache = cache
        
        # Initialize matchers if not provided
        self.text_matcher = text_matcher if text_matcher else TextMatcher(cache=cache)
        self.image_matcher = image_matcher if image_matcher else ImageMatcher(cache=cache)
        
        # Validate weights
        total_weight = text_weight + image_weight
        if not np.isclose(total_weight, 1.0):
            self.logger.warning(
                f"Weights don't sum to 1.0 (text: {text_weight}, image: {image_weight}). "
                f"Normalizing weights."
            )
            self.text_weight = text_weight / total_weight
            self.image_weight = image_weight / total_weight
    
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
    
    def find_matches(
        self, 
        query_product: Product, 
        candidate_products: List[Product],
        max_results: int = 5,
        min_similarity: float = None
    ) -> List[Tuple[Product, float]]:
        """
        Find matching products for the query product.
        
        Args:
            query_product: Product to find matches for
            candidate_products: List of products to search through
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold (default: self.similarity_threshold)
            
        Returns:
            List of tuples (product, similarity) sorted by similarity (highest first)
        """
        if min_similarity is None:
            min_similarity = self.similarity_threshold
            
        matches = []
        
        for candidate in candidate_products:
            # Skip comparing with itself
            if query_product.id == candidate.id:
                continue
                
            # Calculate similarity
            similarity = self.calculate_similarity(query_product, candidate)
            
            # Add to matches if above threshold
            if similarity >= min_similarity:
                matches.append((candidate, similarity))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        return matches[:max_results]
    
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