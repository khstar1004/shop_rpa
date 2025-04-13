# Shop RPA Matching System

이 디렉토리는 상품 매칭을 위한 다양한 매처(Matcher)들을 포함하고 있습니다.

## 주요 기능

### 1. MultiModalMatcher
- 텍스트와 이미지 정보를 결합한 다중 모달 매칭
- 적응형 가중치 시스템으로 상황에 맞는 최적의 매칭 제공
- 카테고리 기반 필터링 및 가격 차이 분석

#### 주요 설정
```python
matcher = MultiModalMatcher(
    text_matcher=TextMatcher(),
    image_matcher=ImageMatcher(),
    text_weight=0.6,  # 텍스트 가중치
    image_weight=0.4,  # 이미지 가중치
    similarity_threshold=0.7,  # 최소 유사도 임계값
    min_price_diff_percent=10.0,  # 최소 가격 차이 퍼센트
    use_adaptive_weights=True  # 적응형 가중치 사용 여부
)
```

### 2. ImageMatcher
- 고급 이미지 매칭 알고리즘 (SIFT, AKAZE, 딥러닝 등)
- 배경 제거 및 이미지 전처리 기능
- 메모리 최적화된 이미지 처리

#### 주요 기능
- 이미지 해시 기반 빠른 매칭
- 딥러닝 기반 특징 추출
- 색상 히스토그램 분석
- SIFT+RANSAC 기반 정밀 매칭

### 3. TextMatcher
- 한국어 특화 텍스트 매칭
- 브랜드 별칭 처리
- 제품 규격 추출 및 비교

#### 주요 기능
- KoSBERT 기반 텍스트 임베딩
- TF-IDF 기반 유사도 계산
- 브랜드 별칭 정규화
- 제품 규격 자동 추출

## 사용 예시

```python
from core.matching import MultiModalMatcher, TextMatcher, ImageMatcher

# 매처 초기화
text_matcher = TextMatcher()
image_matcher = ImageMatcher()
multimodal_matcher = MultiModalMatcher(text_matcher, image_matcher)

# 상품 매칭
matches = multimodal_matcher.find_matches(
    source_product=product1,
    candidate_products=[product2, product3],
    min_text_similarity=0.5,
    min_image_similarity=0.3
)
```

## 성능 최적화

1. **캐싱 전략**
   - 이미지 처리 결과 캐싱
   - 텍스트 임베딩 캐싱
   - 카테고리 호환성 결과 캐싱

2. **메모리 관리**
   - 큰 이미지 자동 크기 조정
   - 동적 캐시 TTL 설정
   - 불필요한 중간 결과물 즉시 해제

3. **병렬 처리**
   - 이미지 다운로드 병렬화
   - 유사도 계산 병렬화

## 주의사항

1. 이미지 매칭 시 메모리 사용량이 높을 수 있으므로, 큰 이미지는 자동으로 크기가 조정됩니다.
2. 텍스트 매칭은 한국어에 최적화되어 있으며, 다른 언어의 경우 성능이 저하될 수 있습니다.
3. 캐시를 사용할 경우 디스크 공간이 필요할 수 있습니다.

## 버전 정보

- 현재 버전: 1.0.0
- 최종 업데이트: 2024-04-12 