# 매칭 모듈

이 디렉토리는 제품 간의 유사도를 비교하고 매칭하는 알고리즘을 포함합니다.

## 구성 요소

- `text_matcher.py`: 텍스트 기반 제품 설명 및 제목 유사도 비교
- `image_matcher.py`: 이미지 기반 제품 유사도 비교

## 텍스트 매처 (text_matcher.py)

텍스트 유사도 비교 기능:

- `TextMatcher` 클래스: 텍스트 기반 유사도 계산
- `calculate_similarity()`: 두 제품 간의 텍스트 유사도 계산
- `preprocess_text()`: 텍스트 전처리 (불용어 제거, 정규화 등)
- `compute_similarity()`: 다양한 알고리즘을 사용하여 유사도 점수 계산

### 사용되는 알고리즘

- TF-IDF 벡터화: 텍스트를 벡터로 변환하여 코사인 유사도 계산
- 자카드 유사도: 두 텍스트의 단어 집합 간의 유사도 계산
- 레벤슈타인 거리: 문자열 간의 편집 거리를 기반으로 한 유사도

## 이미지 매처 (image_matcher.py)

이미지 유사도 비교 기능:

- `ImageMatcher` 클래스: 이미지 기반 유사도 계산
- `calculate_similarity()`: 두 제품 이미지 간의 유사도 계산
- `download_image()`: URL에서 이미지 다운로드
- `extract_features()`: 이미지에서 특징 추출
- `compare_features()`: 추출된 특징 간의 유사도 계산

### 사용되는 알고리즘

- 히스토그램 비교: 이미지 색상 분포 비교
- 특징점 매칭: SIFT/ORB 등의 알고리즘을 사용하여 이미지 특징점 매칭
- 딥러닝 임베딩: 사전 훈련된 CNN을 사용하여 이미지 임베딩 추출 및 비교

## 통합 유사도 계산

텍스트와 이미지 유사도를 결합하여 총체적인 제품 유사도를 계산:

```python
# 가중치 적용 예시
text_weight = 0.7
image_weight = 0.3
combined_similarity = (text_similarity * text_weight) + (image_similarity * image_weight)
```

## 사용 예시

```python
from core.matching.text_matcher import TextMatcher
from core.matching.image_matcher import ImageMatcher
from core.data_models import Product

# 텍스트 매칭
text_matcher = TextMatcher()
source_product = Product(name="프리미엄 스테인리스 텀블러 500ml", ...)
target_product = Product(name="고급 스테인리스 보온 텀블러 500ml", ...)

text_similarity = text_matcher.calculate_similarity(source_product, target_product)
print(f"텍스트 유사도: {text_similarity:.2f}")

# 이미지 매칭
image_matcher = ImageMatcher()
image_similarity = image_matcher.calculate_similarity(source_product, target_product)
print(f"이미지 유사도: {image_similarity:.2f}")

# 통합 유사도
combined_similarity = (text_similarity * 0.7) + (image_similarity * 0.3)
print(f"통합 유사도: {combined_similarity:.2f}")
``` 