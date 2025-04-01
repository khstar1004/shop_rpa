import re
from typing import List, Tuple, Optional
import numpy as np
from Levenshtein import ratio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.caching import FileCache, cache_result

class TextMatcher:
    def __init__(self, cache: Optional[FileCache] = None):
        # Initialize BERT model for Korean text
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        self.cache = cache
        
        # Common Korean brand names and their variations
        self.brand_aliases = {
            '삼성': ['samsung', '샘숭'],
            'LG': ['엘지', '엘쥐'],
            '애플': ['apple', '아이폰'],
            # Add more brand aliases as needed
        }
        
        # Common product type indicators
        self.product_types = [
            '세트', '패키지', '번들', '묶음',
            '단품', '낱개', '개별',
            '리필', '교체', '충전',
            # Add more product type indicators as needed
        ]
        
        # Compile regex patterns
        self.number_pattern = re.compile(r'\d+')
        self.special_char_pattern = re.compile(r'[^\w\s]')
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        if self.cache:
            # Wrap the core logic in a cached method
            return self._cached_calculate_similarity(text1, text2)
        else:
            return self._calculate_similarity_logic(text1, text2)
    
    def _cached_calculate_similarity(self, text1: str, text2: str) -> float:
        # Use a wrapper function to apply the decorator easily
        @cache_result(self.cache, key_prefix="text_sim")
        def cached_logic(t1, t2):
            return self._calculate_similarity_logic(t1, t2)
        
        # Ensure consistent key order regardless of input order
        if text1 > text2:
            text1, text2 = text2, text1
            
        return cached_logic(text1, text2)
    
    def _calculate_similarity_logic(self, text1: str, text2: str) -> float:
        """Core logic for calculating text similarity"""
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        leven_sim = ratio(norm1, norm2)
        
        # Cache embeddings separately for potential reuse
        emb1 = self._get_embedding(norm1)
        emb2 = self._get_embedding(norm2)
        
        bert_sim = float(cosine_similarity([emb1], [emb2])[0, 0])
        
        combined_sim = 0.3 * leven_sim + 0.7 * bert_sim
        return combined_sim
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding, using cache if available."""
        if self.cache:
            cache_key = f"text_embedding|{text}"
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding
            
            embedding = self.model.encode([text])[0]
            self.cache.set(cache_key, embedding)
            return embedding
        else:
            return self.model.encode([text])[0]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace brand variations
        for brand, aliases in self.brand_aliases.items():
            for alias in aliases:
                text = text.replace(alias, brand)
        
        # Remove special characters
        text = self.special_char_pattern.sub(' ', text)
        
        # Normalize numbers (replace sequences of digits with #)
        text = self.number_pattern.sub('#', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from product text"""
        # Normalize text first
        text = self._normalize_text(text)
        
        # Split into terms
        terms = text.split()
        
        # Filter out common product type indicators
        terms = [t for t in terms if t not in self.product_types]
        
        return terms
    
    def find_number_patterns(self, text: str) -> List[Tuple[str, int]]:
        """Find and extract number patterns (e.g., sizes, quantities)"""
        matches = []
        
        # Find all numbers with their context
        for match in self.number_pattern.finditer(text):
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end]
            number = int(match.group())
            matches.append((context, number))
        
        return matches 