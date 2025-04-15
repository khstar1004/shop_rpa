import logging
from typing import Dict, Optional

from core.matching.base_matcher import BaseMatcher


class PriceMatcher(BaseMatcher):
    """Price-based product matching implementation"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.matching_settings = config.get("matching", {})
        self.price_similarity_threshold = float(self.matching_settings.get("price_similarity_threshold", 0.15))
        self.price_weight = float(self.matching_settings.get("price_weight", 0.2))
        
    def match(self, source_price: float, target_price: float) -> float:
        """Calculate price similarity score between source and target prices"""
        try:
            # Calculate price difference percentage
            price_diff = abs(source_price - target_price) / max(source_price, target_price)
            
            # Convert to similarity score (1 - difference)
            similarity = 1.0 - price_diff
            
            # Apply threshold and weight
            if similarity >= self.price_similarity_threshold:
                return similarity * self.price_weight
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error in price matching: {str(e)}")
            return 0.0 