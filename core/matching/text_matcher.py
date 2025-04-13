import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from konlpy.tag import Okt
from Levenshtein import ratio
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.caching import FileCache, cache_result

# Add transformers for tokenization support
try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TextMatcher:
    def __init__(self, cache: Optional[FileCache] = None, use_ko_sbert: bool = True):
        # 로거 초기화
        self.logger = logging.getLogger(__name__)

        # 한국어 특화 모델 사용 여부 선택
        self.use_ko_sbert = use_ko_sbert

        # 한국어에 특화된 Ko-SBERT 모델 (Ko-SRoBERTa-multitask)
        if self.use_ko_sbert:
            try:
                self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")
                self.logger.info("Using Ko-SBERT model: jhgan/ko-sroberta-multitask")

                # Initialize tokenizer if transformers available
                if TRANSFORMERS_AVAILABLE:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            "jhgan/ko-sroberta-multitask"
                        )
                        self.logger.info(
                            "Using ko-sroberta tokenizer for enhanced text processing"
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to load tokenizer: {e}")
                        self.tokenizer = None
                else:
                    self.tokenizer = None
            except Exception as e:
                # 모델 로드 실패시 다국어 모델로 폴백
                self.model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
                self.use_ko_sbert = False
                self.logger.warning(
                    f"Failed to load Ko-SBERT, falling back to multilingual model: {e}"
                )
                self.tokenizer = None
        else:
            # 기존 다국어 모델 (영어, 한국어 둘 다 사용 가능)
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.tokenizer = None

        self.cache = cache

        # TF-IDF 벡터라이저 추가
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.9
        )

        # 한국어 형태소 분석기 추가
        self.okt = Okt()

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
        # 한국어 형태소 분석을 통한 정규화
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # 레벤슈타인 유사도 계산 (20%)
        leven_sim = ratio(norm1, norm2)

        # TF-IDF 유사도 계산 (20%)
        tfidf_sim = self._calculate_tfidf_similarity(norm1, norm2)

        # 토큰화 기반 유사도 계산 (고급 토큰화 사용 시)
        token_sim = 0.0
        if self.tokenizer:
            token_sim = self._calculate_token_similarity(text1, text2)
            # 토큰 유사도가 있으면 TF-IDF 가중치 조정
            tfidf_sim = 0.7 * tfidf_sim + 0.3 * token_sim

        # SBERT 임베딩 유사도 계산 (60%) - 한국어 특화 모델 적용으로 가중치 증가
        # Cache embeddings separately for potential reuse
        emb1 = self._get_embedding(norm1)
        emb2 = self._get_embedding(norm2)

        # 코사인 유사도 계산 (PyTorch 텐서 활용)
        if self.use_ko_sbert:
            # Ko-SBERT 모델에 최적화된 유사도 계산 (PyTorch util 활용)
            emb1_tensor = util.normalize_embeddings(torch.tensor([emb1]))
            emb2_tensor = util.normalize_embeddings(torch.tensor([emb2]))
            bert_sim = float(util.pytorch_cos_sim(emb1_tensor, emb2_tensor).item())
        else:
            # 일반 코사인 유사도 계산 (numpy 사용)
            bert_sim = float(cosine_similarity([emb1], [emb2])[0, 0])

        # 규격 유사도 계산 (추가 가중치)
        spec_sim = self._calculate_spec_similarity(text1, text2)

        # 결합 유사도 계산 (가중치 조정)
        combined_sim = 0.2 * leven_sim + 0.2 * tfidf_sim + 0.6 * bert_sim

        # 규격이 일치하면 유사도 보너스 부여
        if spec_sim > 0.8:
            combined_sim = min(1.0, combined_sim * 1.1)

        return combined_sim

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

    def _get_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 계산 (캐시 활용)"""
        if self.cache:
            cache_key = (
                f"text_embedding|{text}|{'ko_sbert' if self.use_ko_sbert else 'multi'}"
            )
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

            embedding = self.model.encode(text, convert_to_numpy=True)
            self.cache.set(cache_key, embedding)
            return embedding
        else:
            return self.model.encode(text, convert_to_numpy=True)

    def _normalize_text(self, text: str) -> str:
        """한국어 특화 텍스트 정규화"""
        # 소문자 변환
        text = text.lower()

        # 브랜드 변형 치환 (최적화된 순서)
        for brand, aliases in self.brand_aliases.items():
            # 가장 긴 별칭부터 처리 (부분 매칭 방지)
            sorted_aliases = sorted(aliases, key=len, reverse=True)
            for alias in sorted_aliases:
                if alias in text:
                    text = text.replace(alias, brand)
                    break  # 한 번만 치환하고 다음 브랜드로

        # 한국어 형태소 분석을 통한 정규화
        try:
            # 명사, 형용사, 동사만 추출
            pos_tags = self.okt.pos(text, norm=True, stem=True)
            filtered_words = [
                word for word, pos in pos_tags if pos in ["Noun", "Adjective", "Verb"]
            ]

            # 불용어 제거 및 정규화된 텍스트 재구성
            if filtered_words:
                text = " ".join(filtered_words)
        except Exception as e:
            self.logger.warning(f"형태소 분석 실패: {e}")
            # 형태소 분석 오류 시 기존 처리 방식 사용
            pass

        # 특수 문자 제거
        text = self.special_char_pattern.sub(" ", text)

        # 숫자 정규화 (숫자 시퀀스를 #으로 치환)
        text = self.number_pattern.sub("#", text)

        # 과도한 공백 제거
        text = " ".join(text.split())

        return text

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
