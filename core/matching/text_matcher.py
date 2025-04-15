import logging
import re
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from konlpy.tag import Okt
from Levenshtein import ratio
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.caching import FileCache, cache_result
from core.matching.base_matcher import BaseMatcher
from core.data_models import Product, MatchResult

# Add transformers for tokenization support
try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TextMatcher(BaseMatcher):
    """Text-based product matching implementation"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None, cache: Optional[FileCache] = None):
        # Initialize parent class first
        super().__init__(config, logger)
        
        # Get text settings with default values
        self.text_settings = config.get("TEXT_MATCHER", {}) if isinstance(config, dict) else {}
        self.cache = cache
        
        # Set default values for settings
        self.similarity_threshold = float(self.text_settings.get("similarity_threshold", 0.75))
        self.text_similarity_threshold = float(self.text_settings.get("text_similarity_threshold", 0.75))
        self.text_weight = float(self.text_settings.get("text_weight", 0.7))
        self.use_ko_sbert = bool(self.text_settings.get("use_ko_sbert", True))
        self.use_stemming = bool(self.text_settings.get("use_stemming", False))

        # Model and Tokenizer Initialization
        self.model = None
        self.tokenizer = None
        self._load_models()

        # TF-IDF Vectorizer
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,
                ngram_range=(1, 2)
            )
        except Exception as e:
            self.logger.error(f"TF-IDF Vectorizer initialization failed: {e}")
            self.tfidf_vectorizer = None

        # Korean Morphological Analyzer (Okt)
        try:
            self.okt = Okt()
            # Test Okt initialization
            _ = self.okt.morphs("테스트")
            self.logger.info("Okt initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Okt: {e}. Check Java/JDK installation.", exc_info=True)
            self.okt = None

        # Brand aliases 확장
        self.brand_aliases = {
            "삼성": ["samsung", "샘숭", "samseong", "삼성전자"],
            "LG": ["엘지", "엘쥐", "lg전자", "엘지전자"],
            "애플": [
                "apple",
                "아이폰",
                "iphone",
                "맥북",
                "macbook",
                "아이패드",
                "ipad",
            ],
            "현대": ["hyundai", "현대자동차", "현대차"],
            "기아": ["kia", "기아자동차", "기아차"],
            "쿠팡": ["coupang", "쿠팡로켓배송"],
            "다이슨": ["dyson", "다이손"],
            "브라운": ["braun", "브라운전자"],
            "샤오미": ["xiaomi", "mi", "미"],
            "아이닉": ["ionic", "아이오닉"],
            "마이크로소프트": ["microsoft", "ms", "엠에스"],
            "나이키": ["nike"],
            "아디다스": ["adidas"],
            "퓨마": ["puma"],
            "필립스": ["philips", "필립스전자"],
            "일렉트로룩스": ["electrolux", "일렉트로럭스"],
            # Add more brand aliases as needed
        }

        # 제품 종류 확장
        self.product_types = [
            "세트",
            "패키지",
            "번들",
            "묶음",
            "단품",
            "낱개",
            "개별",
            "리필",
            "교체",
            "충전",
            "최신형",
            "신형",
            "구형",
            "이전",
            "한정판",
            "스페셜",
            "특별판",
            "에디션",
            "기획",
            "할인",
            "프로모션",
            "이벤트",
            "대용량",
            "소용량",
            "미니",
            "점보",
            "프리미엄",
            "고급",
            "스탠다드",
            "기본형",
            # Add more product type indicators as needed
        ]

        # 규격 및 스펙 패턴 추가
        self.spec_patterns = {
            "size": re.compile(
                r"(\d+[\s]*[x×][\s]*\d+[\s]*[x×]?[\s]*\d*[\s]*(?:mm|cm|m)?)",
                re.IGNORECASE,
            ),
            "weight": re.compile(r"(\d+[\s]*(?:kg|g|t|mg))", re.IGNORECASE),
            "volume": re.compile(r"(\d+[\s]*(?:ml|l|리터|oz|온스))", re.IGNORECASE),
            "screen": re.compile(
                r'(\d+(?:\.\d+)?[\s]*(?:인치|inch|"|\'))', re.IGNORECASE
            ),
        }

        # 숫자 패턴 및 특수문자 패턴은 유지
        self.number_pattern = re.compile(r"\d+")
        self.special_char_pattern = re.compile(r"[^\w\s]")

    def _load_models(self):
        """Loads the SentenceTransformer model and tokenizer."""
        sbert_model_name = "jhgan/ko-sroberta-multitask"
        fallback_model_name = "paraphrase-multilingual-MiniLM-L12-v2"

        if self.use_ko_sbert:
            try:
                self.logger.info(f"Loading Ko-SBERT model: {sbert_model_name}")
                self.model = SentenceTransformer(sbert_model_name)
                self.logger.info("Ko-SBERT model loaded successfully.")

                if TRANSFORMERS_AVAILABLE:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)
                        self.logger.info(f"Using tokenizer from {sbert_model_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load tokenizer for {sbert_model_name}: {e}")
                        self.tokenizer = None
                else:
                    self.tokenizer = None
            except Exception as e:
                self.logger.warning(
                    f"Failed to load Ko-SBERT model ({sbert_model_name}), falling back to {fallback_model_name}: {e}",
                    exc_info=True
                )
                try:
                    self.model = SentenceTransformer(fallback_model_name)
                    self.use_ko_sbert = False
                    self.logger.info(f"Loaded fallback multilingual model: {fallback_model_name}")
                    self.tokenizer = None
                except Exception as fallback_e:
                    self.logger.error(f"Failed to load even the fallback SBERT model: {fallback_e}", exc_info=True)
                    self.model = None
                    self.tokenizer = None
        else:
            try:
                self.logger.info(f"Loading multilingual SBERT model: {fallback_model_name}")
                self.model = SentenceTransformer(fallback_model_name)
                self.logger.info(f"Multilingual model loaded successfully.")
                self.tokenizer = None
            except Exception as e:
                self.logger.error(f"Failed to load multilingual SBERT model: {e}", exc_info=True)
                self.model = None
                self.tokenizer = None

    def match(self, source_product: Product, target_product: Product) -> float:
        """Calculate similarity score between source and target products.
        
        Args:
            source_product: Source product to match from
            target_product: Target product to match against
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        return self.calculate_similarity(source_product.name, target_product.name)

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
            min_image_similarity: Minimum image similarity threshold (ignored for text matcher)
            min_combined_similarity: Minimum combined similarity threshold (ignored for text matcher)
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match results sorted by similarity (descending)
        """
        threshold = min_text_similarity or self.similarity_threshold
        similarities = []
        
        for candidate in candidate_products:
            similarity = self.match(source_product, candidate)
            if similarity >= threshold:
                similarities.append((candidate, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply max_matches limit if specified
        if max_matches is not None:
            similarities = similarities[:max_matches]
        
        # Convert to MatchResult objects
        results = [
            MatchResult(
                source_product=source_product,
                matched_product=candidate,
                similarity_score=similarity,
                match_type="text"
            )
            for candidate, similarity in similarities
        ]
        
        return results

    def find_best_match(
        self,
        source_product: Product,
        candidate_products: List[Product],
        min_text_similarity: float = None,
        min_image_similarity: float = None,
        min_combined_similarity: float = None,
    ) -> Optional[MatchResult]:
        """Find the best matching product among candidates.
        
        Args:
            source_product: Source product to match from
            candidate_products: List of candidate products to match against
            min_text_similarity: Minimum text similarity threshold
            min_image_similarity: Minimum image similarity threshold (ignored for text matcher)
            min_combined_similarity: Minimum combined similarity threshold (ignored for text matcher)
            
        Returns:
            Best match result or None if no match meets the threshold
        """
        matches = self.find_matches(
            source_product,
            candidate_products,
            min_text_similarity=min_text_similarity,
            max_matches=1
        )
        
        return matches[0] if matches else None

    def batch_find_matches(
        self,
        query_products: List[Product],
        candidate_products: List[Product],
        max_results_per_query: int = None,
        min_similarity: float = None,
    ) -> Dict[str, List[Tuple[Product, float]]]:
        """Find matches for multiple query products in batch.
        
        Args:
            query_products: List of products to find matches for
            candidate_products: List of candidate products to match against
            max_results_per_query: Maximum number of matches to return per query
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary mapping query product IDs to lists of (matched product, similarity) tuples
        """
        results = {}
        threshold = min_similarity or self.similarity_threshold
        
        for query in query_products:
            matches = self.find_matches(
                query,
                candidate_products,
                min_text_similarity=threshold,
                max_matches=max_results_per_query
            )
            
            results[query.id] = [(match.matched_product, match.similarity_score) for match in matches]
        
        return results

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculates combined text similarity, using cache if available."""
        if not text1 or not text2:
            return 0.0

        # Ensure consistent key order for caching
        if text1 > text2:
            text1, text2 = text2, text1

        if self.cache:
            # Generate cache key using hashlib
            cache_key_string = f"{text1}|{text2}"
            cache_key = hashlib.md5(cache_key_string.encode()).hexdigest()
            return self._cached_calculate_similarity(text1, text2, cache_key)
        else:
            # Calculate directly if no cache
            return self._calculate_similarity_logic(text1, text2)

    def _cached_calculate_similarity(self, text1: str, text2: str, cache_key: str) -> float:
        """Wrapper for cached similarity calculation using a pre-generated key."""

        # Define the logic function to be cached
        def logic_to_cache(t1, t2):
            return self._calculate_similarity_logic(t1, t2)

        # Apply the cache decorator with the provided key
        cached_logic = cache_result(self.cache, key=cache_key)(logic_to_cache)

        return cached_logic(text1, text2)

    def _calculate_similarity_logic(self, text1: str, text2: str) -> float:
        """Core logic for calculating text similarity"""
        if self.model is None:
            self.logger.error("SBERT model not available, cannot calculate text similarity.")
            return 0.0

        try:
            # Preprocess and normalize
            norm1 = self._normalize_text(text1)
            norm2 = self._normalize_text(text2)

            if not norm1 or not norm2:
                return 0.0

            # --- Similarity Components --- #

            # 1. Levenshtein Similarity (Weight: 0.15)
            leven_sim = ratio(norm1, norm2)

            # 2. TF-IDF Similarity (Weight: 0.15)
            tfidf_sim = self._calculate_tfidf_similarity(norm1, norm2)

            # 3. Token-based Similarity (Optional, Weight: 0.10 if used)
            token_sim = 0.0
            use_token_sim = bool(self.tokenizer)
            if use_token_sim:
                token_sim = self._calculate_token_similarity(text1, text2)

            # 4. SBERT Embedding Similarity (Weight: 0.60 or adjusted)
            emb1 = self._get_embedding(norm1)
            emb2 = self._get_embedding(norm2)
            if emb1 is None or emb2 is None:
                self.logger.warning(f"Could not get embeddings for similarity calculation.")
                return (0.3 * leven_sim + 0.7 * tfidf_sim)

            bert_sim = self._compute_embedding_similarity(emb1, emb2)

            # 5. Specification Similarity (Bonus/Malus)
            spec_sim_score = self._calculate_spec_similarity(text1, text2)

            # --- Combine Similarities --- #
            # Adjust weights based on token similarity availability
            weights = {
                'leven': 0.15,
                'tfidf': 0.15,
                'token': 0.10 if use_token_sim else 0.0,
                'bert': 0.60
            }
            # Renormalize weights if token sim is not used
            if not use_token_sim:
                total_weight = weights['leven'] + weights['tfidf'] + weights['bert']
                weights['leven'] /= total_weight
                weights['tfidf'] /= total_weight
                weights['bert'] /= total_weight

            combined_sim = (
                weights['leven'] * leven_sim +
                weights['tfidf'] * tfidf_sim +
                weights['token'] * token_sim +
                weights['bert'] * bert_sim
            )

            # Apply specification bonus/malus
            if spec_sim_score > 0.8:
                bonus = (spec_sim_score - 0.8) * 0.5
                combined_sim = min(1.0, combined_sim + bonus * combined_sim)
                self.logger.debug(f"Applying spec similarity bonus ({bonus:.2f}) for '{text1[:30]}' vs '{text2[:30]}'")
            elif spec_sim_score < 0.2 and combined_sim > 0.5:
                malus = (0.2 - spec_sim_score) * 0.5
                combined_sim = max(0.0, combined_sim - malus * combined_sim)
                self.logger.debug(f"Applying spec similarity malus ({malus:.2f}) for '{text1[:30]}' vs '{text2[:30]}'")

            # Final clipping
            final_sim = max(0.0, min(1.0, combined_sim))

            self.logger.debug(f"Text similarity details for ('{text1[:30]}...', '{text2[:30]}...'): "
                             f"Leven={leven_sim:.3f}, TFIDF={tfidf_sim:.3f}, Token={token_sim:.3f}, "
                             f"BERT={bert_sim:.3f}, Spec={spec_sim_score:.3f}, Combined={final_sim:.3f}")

            return float(final_sim)

        except Exception as e:
            self.logger.error(f"Error in _calculate_similarity_logic for '{text1[:50]}...' and '{text2[:50]}...': {e}", exc_info=True)
            return 0.0

    def _compute_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two embeddings."""
        try:
            # Ensure embeddings are numpy arrays
            emb1 = np.asarray(emb1)
            emb2 = np.asarray(emb2)

            # Normalize embeddings for cosine similarity calculation (more stable)
            norm_emb1 = emb1 / np.linalg.norm(emb1)
            norm_emb2 = emb2 / np.linalg.norm(emb2)

            # Calculate cosine similarity
            similarity = np.dot(norm_emb1, norm_emb2)

            # Clip similarity to [0, 1] range (cosine sim is [-1, 1])
            return max(0.0, min(1.0, float(similarity)))
        except Exception as e:
            self.logger.error(f"Error computing embedding similarity: {e}", exc_info=True)
            return 0.0

    def _calculate_token_similarity(self, text1: str, text2: str) -> float:
        """토큰화 기반 유사도 계산 (transformers tokenizer 활용)"""
        try:
            # 텍스트 토큰화
            tokens1 = self.tokenizer.tokenize(text1)
            tokens2 = self.tokenizer.tokenize(text2)

            # ## 제거
            tokens1 = [token.replace("##", "") for token in tokens1]
            tokens2 = [token.replace("##", "") for token in tokens2]

            # 공통 토큰 수 계산
            common_tokens = set(tokens1).intersection(set(tokens2))

            # Jaccard 유사도 계산
            if len(set(tokens1).union(set(tokens2))) > 0:
                return len(common_tokens) / len(set(tokens1).union(set(tokens2)))
            else:
                return 0.0
        except Exception as e:
            self.logger.warning(f"Token similarity calculation failed: {e}")
            return 0.0

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF 벡터를 사용한 유사도 계산"""
        try:
            # 두 텍스트로 TF-IDF 벡터라이저 학습
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])

            # 코사인 유사도 계산
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0])
        except Exception:
            # 에러 발생 시 기본값 반환
            return 0.5

    def _calculate_spec_similarity(self, text1: str, text2: str) -> float:
        """제품 규격 유사도 계산"""
        similarities = []

        for spec_type, pattern in self.spec_patterns.items():
            matches1 = pattern.findall(text1)
            matches2 = pattern.findall(text2)

            if matches1 and matches2:
                # 각 규격 타입별로 일치하는 규격이 하나라도 있으면 1.0, 없으면 0.0
                for m1 in matches1:
                    for m2 in matches2:
                        if self._normalize_spec(m1) == self._normalize_spec(m2):
                            similarities.append(1.0)
                            break

        # 규격 유사도가 있으면 평균 계산, 없으면 0.5 반환
        return np.mean(similarities) if similarities else 0.5

    def _normalize_spec(self, spec: str) -> str:
        """규격 문자열 정규화"""
        # 공백 제거 및 소문자 변환
        spec = re.sub(r"\s+", "", spec.lower())
        # 숫자와 단위만 남기기
        spec = re.sub(r"[^0-9a-z×x]", "", spec)
        return spec

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Gets the SBERT embedding for a text, using cache if available."""
        if self.model is None:
            self.logger.error("SBERT model not loaded, cannot get embedding.")
            return None
        if not text:
            return None

        # Define the core embedding logic
        def embedding_logic(t):
            try:
                embedding = self.model.encode(t, convert_to_numpy=True)
                return embedding
            except Exception as e:
                self.logger.error(f"Failed to encode text for embedding: '{t[:100]}...': {e}", exc_info=True)
                return None

        if self.cache:
            # Use cache if available
            cache_key = f"sbert_emb_{hashlib.md5(text.encode()).hexdigest()}"
            # Assuming cache can handle numpy arrays
            cached_embedding = cache_result(self.cache, key=cache_key)(embedding_logic)
            return cached_embedding(text)
        else:
            # Calculate directly if no cache
            return embedding_logic(text)

    def _normalize_text(self, text: str) -> str:
        """Normalize text by lowercasing, removing special chars, and optionally stemming/morph analysis."""
        if not text:
            return ""
        try:
            # Lowercase
            text = text.lower()

            # Replace brand aliases
            for brand, aliases in self.brand_aliases.items():
                for alias in aliases:
                    # Use word boundaries to avoid partial matches like 'samsung' in 'samsungelectronics'
                    text = re.sub(r'\b' + re.escape(alias) + r'\b', brand, text)

            # Remove special characters (keep spaces, alphanumeric, Hangul)
            # Improved regex to keep relevant characters
            text = re.sub(r'[^\w\s가-힣一-龥]', ' ', text)

            # Optional: Korean Morphological Analysis or Stemming
            if self.okt:
                # Use Okt for more accurate Korean tokenization/normalization
                # Consider using norm() or stem=True based on desired level of normalization
                morphs = self.okt.morphs(text, norm=True, stem=self.use_stemming)
                text = ' '.join(morphs)
            elif self.use_stemming:
                 # Basic stemming (less accurate for Korean) - Placeholder if needed
                 self.logger.warning("Basic stemming requested but Okt not available. Stemming might be inaccurate.")
                 # Add a basic stemming logic if required, e.g., using a simple stemmer
                 pass

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            self.logger.error(f"Error normalizing text: '{text[:100]}...': {e}", exc_info=True)
            return text # Return original text on error

    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from product text"""
        # 텍스트 정규화
        text = self._normalize_text(text)

        # 고급 토큰화 사용 시 토큰화 결과 활용
        if self.tokenizer:
            try:
                tokens = self.tokenizer.tokenize(text)
                tokens = [token.replace("##", "") for token in tokens]
                # 제품 타입 필터링
                tokens = [t for t in tokens if t not in self.product_types]
                return tokens
            except:
                pass

        # 일반 텍스트 분리 방식 (토큰화 실패 시 폴백)
        terms = text.split()

        # 제품 타입 표시자 필터링
        terms = [t for t in terms if t not in self.product_types]

        return terms

    def find_number_patterns(self, text: str) -> List[Tuple[str, int]]:
        """Find and extract number patterns (e.g., sizes, quantities)"""
        matches = []

        # 모든 숫자와 컨텍스트 찾기
        for match in self.number_pattern.finditer(text):
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end]
            number = int(match.group())
            matches.append((context, number))

        return matches

    def extract_product_specs(self, text: str) -> Dict[str, List[str]]:
        """제품 규격 정보 추출"""
        specs = {}

        for spec_type, pattern in self.spec_patterns.items():
            matches = pattern.findall(text)
            if matches:
                specs[spec_type] = matches

        return specs

    def calculate_batch_similarity(
        self, main_text: str, comparison_texts: List[str]
    ) -> List[float]:
        """여러 텍스트와의 유사도를 일괄 계산 (koSBERT 방식)"""
        if not comparison_texts:
            return []

        # 텍스트 정규화
        norm_main = self._normalize_text(main_text)
        norm_texts = [self._normalize_text(text) for text in comparison_texts]

        # 메인 텍스트 임베딩 계산
        main_embedding = self._get_embedding(norm_main)

        similarities = []

        # 각 비교 텍스트와의 유사도 계산
        for norm_text in norm_texts:
            text_embedding = self._get_embedding(norm_text)

            if self.use_ko_sbert:
                # Ko-SBERT 모델에 최적화된 유사도 계산
                main_tensor = util.normalize_embeddings(torch.tensor([main_embedding]))
                text_tensor = util.normalize_embeddings(torch.tensor([text_embedding]))
                similarity = float(
                    util.pytorch_cos_sim(main_tensor, text_tensor).item()
                )
            else:
                # 일반 코사인 유사도 계산
                similarity = float(
                    cosine_similarity([main_embedding], [text_embedding])[0, 0]
                )

            similarities.append(similarity)

        return similarities

    def tokenize_product_names(self, product_names: List[str]) -> List[List[str]]:
        """제품명 토큰화 (tokenize_product_names.py 기능 통합)"""
        if not self.tokenizer:
            self.logger.warning("Tokenizer not available for product name tokenization")
            return [text.split() for text in product_names]

        try:
            # 제품명 토큰화
            tokenized_names = [self.tokenizer.tokenize(name) for name in product_names]

            # "##" 제거
            tokenized_names = [
                [token.replace("##", "") for token in name] for name in tokenized_names
            ]

            return tokenized_names
        except Exception as e:
            self.logger.error(f"Product name tokenization failed: {e}")
            # 실패 시 간단한 공백 기반 분리로 폴백
            return [text.split() for text in product_names]
