# 유틸리티 모듈

이 디렉토리는 RPA 시스템 전반에서 사용되는 유틸리티 기능을 포함합니다.

## 구성 요소

- `reporting.py`: 분석 결과 리포트 생성
- `caching.py`: 데이터 캐싱 및 재사용 기능
- `config.py`: 설정 파일 로드 및 관리

## 리포팅 모듈 (reporting.py)

보고서 생성 기능을 제공합니다:

- `generate_primary_report()`: 1차 상세 보고서 생성 (모든 제품 데이터)
  - 매칭된 모든 상품 정보를 포함
  - 가격 비교 및 유사도 정보 제공
  - 오류 정보 기록
  - 조건부 서식 적용 (음수 가격 차이에 노란색 하이라이트)

- `generate_secondary_report()`: 2차 필터링된 보고서 생성 (특정 조건에 맞는 제품만)
  - 필터링 규칙 적용:
    - 고려기프트나 네이버 상품이 원본 상품보다 저렴한 경우만 포함
    - 프로모션 사이트는 1% 이상 저렴한 경우만 포함
    - 일반 쇼핑몰은 10% 이상 저렴한 경우만 포함
  - 필터링된 상품만 포함한 간결한 보고서 생성
  - 조건부 서식 적용

두 보고서 모두 Excel 파일로 생성되며, 열 너비 자동 조정 및 헤더 고정이 적용됩니다.

## 캐싱 기능 (caching.py)

데이터 캐싱 및 재사용 기능:

- `ImageCache`: 이미지 다운로드 및 특징 추출 결과 캐싱
- `SearchResultCache`: API 검색 결과 캐싱
- `ProductCache`: 제품 데이터 캐싱

## 설정 관리 (config.py)

설정 파일 로드 및 관리:

- `load_config()`: config.ini 파일에서 설정 로드
- `get_config_value()`: 특정 설정 값 가져오기
- `save_config()`: 수정된 설정 저장

### 주요 설정 항목

- API 키 및 인증 정보
- 경로 및 디렉토리 설정
- 유사도 임계값 설정
- 필터링 규칙 파라미터

## 사용 예시

```python
# 설정 로드
from utils.config import load_config
config = load_config()

# 캐싱 사용
from utils.caching import ImageCache
cache = ImageCache(config['PATHS']['CACHE_DIR'])
img_features = cache.get_or_compute('http://example.com/image.jpg', compute_function)

# 리포트 생성
from utils.reporting import generate_primary_report, generate_secondary_report
from datetime import datetime

start_time = datetime.now()
# ... 처리 로직 ...
end_time = datetime.now()

primary_report = generate_primary_report(results, config, start_time, end_time)
secondary_report = generate_secondary_report(results, config, input_filepath)
``` 