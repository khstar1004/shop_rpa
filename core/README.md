# Core 모듈

이 디렉토리는 쇼핑 RPA 시스템의 핵심 기능을 담당하는 모듈들을 포함합니다.

## 구조

- `data_models.py`: 데이터 모델 및 구조체 정의
- `processing.py`: 파일 처리, 제품 매칭 및 보고서 생성 로직
- `scraping/`: 외부 사이트에서 데이터를 수집하는 모듈들
- `matching/`: 텍스트 및 이미지 유사도 비교 알고리즘

## 주요 흐름

1. `processing.py`에서 엑셀 파일을 로드하고 각 행을 `Product` 객체로 변환
2. 각 `Product`에 대해 고려기프트와 네이버 쇼핑에서 매칭되는 상품 검색
3. 텍스트 및 이미지 유사도 계산을 통해 최적의 매치 선택
4. 매칭 결과에 따라 가격 차이 계산 및 보고서 생성

## 데이터 모델 (data_models.py)

핵심 데이터 모델:

- `Product`: 상품 정보를 담는 기본 클래스
- `ProcessingResult`: 단일 제품의 처리 결과
- `MatchResult`: 제품 매칭 결과와 유사도 점수

## 처리 로직 (processing.py)

- `process_file()`: 엑셀 파일을 처리하고 결과 보고서 생성
- `process_product()`: 개별 제품에 대한 매칭 처리
- `_calculate_match_similarities()`: 매칭된 제품 간의 유사도 점수 계산
- `_create_product_from_row()`: 엑셀 행에서 `Product` 객체 생성
- `_generate_reports()`: 1차 및 2차 보고서 생성

## 사용 예시

```python
from core.processing import process_file
from utils.config import load_config

config = load_config()
input_file = "data/input/products.xlsx"
primary_report, secondary_report = process_file(input_file, config)
``` 