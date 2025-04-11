# 유틸리티 모듈

이 디렉토리는 Shop RPA 시스템의 여러 부분에서 공통적으로 사용되는 보조 기능 및 유틸리티 함수들을 포함합니다. 특정 비즈니스 로직이나 사용자 인터페이스에 직접적으로 종속되지 않는 재사용 가능한 코드들로 구성됩니다.

## 구성 요소

- **`excel_utils.py`**: Microsoft Excel 파일(.xlsx, .xls)과 관련된 다양한 작업을 수행하는 함수들을 제공합니다. 파일 읽기, 쓰기, 특정 셀 서식 지정(색상, 테두리, 정렬), 데이터 유효성 검사, 이미지 삽입 등의 기능을 포함할 수 있습니다. (`reporting.py`와 기능적으로 일부 겹칠 수 있으므로 역할 분담 확인 필요)
- **`reporting.py`**: 처리된 데이터를 바탕으로 최종 사용자 또는 분석을 위한 보고서를 생성하는 기능을 담당합니다. 주로 Excel 형식을 다루며, 데이터 요약, 특정 조건에 따른 필터링, 조건부 서식 적용(예: 가격 차이 강조), 차트 삽입 등의 기능을 제공합니다.
- **`caching.py`**: 반복적인 계산이나 외부 API 호출 결과를 저장하여 성능을 향상시키는 캐싱 메커니즘을 제공합니다. 이미지 특징 벡터, 웹 스크래핑 결과, API 응답 등을 파일 시스템이나 메모리에 임시 저장하고 재사용하는 클래스(예: `ImageCache`, `SearchResultCache`)를 포함할 수 있습니다.
- **`config.py`**: 애플리케이션의 설정을 외부 파일(예: `config.ini`)에서 읽고 관리하는 기능을 제공합니다. API 키, 파일 경로, 동작 제어 파라미터(임계값 등)를 코드와 분리하여 관리할 수 있게 합니다. 설정을 읽고, 특정 값을 조회하고, 필요한 경우 설정을 저장하는 함수(`load_config`, `get_config_value`, `save_config`)를 포함합니다.
- **`preprocessing.py`**: 입력 데이터(주로 텍스트 또는 테이블 형태)를 처리 파이프라인에 적합한 형태로 변환하는 전처리 함수들을 포함합니다. 데이터 클리닝(불필요한 문자 제거, 공백 정리), 형식 변환(숫자, 날짜), 정규화, 특정 규칙 기반의 데이터 수정 등의 작업을 수행할 수 있습니다.

## 주요 기능 상세

- **리포팅 (`reporting.py`)**: 상세 및 요약 보고서 자동 생성, 사용자 정의 가능한 필터링 규칙 적용, Excel 서식 자동화 (열 너비, 헤더 고정, 조건부 서식).
- **캐싱 (`caching.py`)**: 다양한 데이터 타입(이미지, 검색 결과, 제품 정보)에 대한 캐시 관리, 캐시 유효 기간 설정 가능성.
- **설정 관리 (`config.py`)**: `config.ini` 파일 기반 설정 로드/저장, 섹션별 설정 값 접근.
- **Excel 처리 (`excel_utils.py`)**: Pandas DataFrame과 Openpyxl 라이브러리를 활용한 저수준/고수준 Excel 조작 기능.
- **데이터 전처리 (`preprocessing.py`)**: 문자열 처리(정규식 활용), 데이터 타입 변환, 결측치 처리 등 일반적인 데이터 준비 작업.

## 사용 예시

```python
# 설정 로드
from utils.config import load_config
config = load_config()
api_key = config.get('API', 'naver_api_key', fallback=None)

# 캐싱 사용
from utils.caching import ImageCache
from some_module import compute_image_features # 예시 함수

cache_dir = config.get('PATHS', 'cache_directory')
image_cache = ImageCache(cache_dir)
img_features = image_cache.get_or_compute('http://example.com/image.jpg', compute_image_features)

# Excel 파일 읽기 (예시)
from utils.excel_utils import read_excel_data

data = read_excel_data("input.xlsx", sheet_name="Sheet1")

# 데이터 전처리 (예시)
from utils.preprocessing import clean_text

cleaned_name = clean_text("  [특가] Awesome Product!!  ")

# 리포트 생성 (reporting 모듈 사용은 reporting.py 내부 로직에 따라 다름)
# from utils.reporting import generate_final_report
# report_path = generate_final_report(processed_data, config)
``` 