# Shop RPA System

상품 가격 비교 자동화 시스템 (Product Price Comparison Automation System)

## 개요

본 프로젝트는 해오름기프트 상품과 경쟁사(고려기프트, 네이버쇼핑) 상품을 자동으로 매칭하고, 가격을 비교하여 경쟁력 있는 가격 정보를 제공하는 RPA(Robotic Process Automation) 시스템입니다. 텍스트 및 이미지 유사도 분석을 통해 정확한 상품 매칭을 수행하고, 가격 차이를 분석하여 상세 보고서를 생성합니다.

## 주요 기능

- 본사 상품 코드 & 네이버 쇼핑 검색 결과 비교
- 고려기프트 사이트 검색 & 상품 비교
- 텍스트 유사도 기반 상품 매칭
- 이미지 유사도 기반 상품 매칭
- 프로모션 사이트 자동 분류 (네이버 쇼핑)
- 캐싱을 통한 성능 최적화
- 로컬 파일 기반 처리 (엑셀 입력/출력)
- 두 가지 유형의 보고서 생성:
  - 1차 보고서: 모든 상품 분석 결과 (매칭 상세 정보 포함)
  - 2차 보고서: 특정 가격 차이 조건에 맞는 상품만 필터링한 결과

## 기능 (Features)

- 해오름기프트 상품과 고려기프트/네이버쇼핑 상품 자동 매칭
- 텍스트 및 이미지 기반 상품 유사도 분석
- 실시간 가격 정보 수집 및 비교
- 프로모션 사이트 자동 식별 및 필터링
- 가격 차이 보고서 자동 생성 (1차/2차 Excel 보고서)
- 상세 필터링 로직 적용 (가격 차이 비율에 따른 필터링)
- 사용자 친화적 GUI 인터페이스

## 고도화 내용

1. **데이터 모델 개선**
   - 원본 데이터 보존 기능 추가
   - 오류 처리 및 로깅 강화

2. **네이버 크롤링 개선**
   - 네이버 API 대신 웹 크롤링 방식으로 변경
   - 프로모션 사이트 자동 식별 기능 추가
   - 키워드 기반 필터링 로직 구현

3. **보고서 생성 기능 강화**
   - 1차/2차 보고서 분리 및 개선
   - 복잡한 필터링 규칙 적용
   - 엑셀 포맷팅 개선 (조건부 서식, 숫자 형식 등)

4. **문서화 개선**
   - 각 모듈별 README 파일 추가
   - 코드 주석 및 설명 강화

## 시스템 요구사항 (System Requirements)

- Python 3.8 이상
- 32GB RAM 이상 권장
- NVIDIA GPU (선택사항, 이미지 유사도 분석 가속)
- Windows 10 이상

## 설치 방법 (Installation)

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. 의존성 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 설정:
- `config.ini` 파일에서 필요한 설정 수정

## 실행 방법 (Usage)

```bash
python main.py
```

GUI 인터페이스가 실행되면 다음 단계를 따릅니다:

1. 입력 파일 선택 (Excel)
2. 설정 확인 (필요시 조정)
3. 처리 시작 버튼 클릭
4. 진행 상황 모니터링
5. 결과 보고서 확인 (1차/2차)

## 주의사항 (Notes)

- 웹 크롤링 시 서버 부하를 고려하여 딜레이 적용됨
- 이미지 처리를 위한 충분한 저장 공간 확보
- 안정적인 인터넷 연결 필요

## 프로젝트 구조

```
Shop_RPA/
│
├── main.py                # 프로그램 진입점
├── config.ini            # 설정 파일
├── requirements.txt      # 의존성 패키지 목록
├── README.md            # 프로젝트 문서
│
├── gui/                  # GUI 관련 모듈
│   ├── __init__.py
│   ├── README.md         # GUI 모듈 설명
│   └── main_window.py    # 메인 윈도우 클래스
│
├── core/                 # 핵심 기능 모듈
│   ├── __init__.py
│   ├── README.md         # 코어 모듈 설명
│   ├── data_models.py    # 데이터 모델 클래스
│   ├── processing.py     # 메인 처리 로직
│   │
│   ├── matching/         # 매칭 관련 모듈
│   │   ├── __init__.py
│   │   ├── README.md     # 매칭 모듈 설명
│   │   ├── text_matcher.py
│   │   └── image_matcher.py
│   │
│   └── scraping/         # 웹 스크래핑 모듈
│       ├── __init__.py
│       ├── README.md     # 스크래핑 모듈 설명
│       ├── koryo_scraper.py
│       └── naver_crawler.py
│
├── utils/                # 유틸리티 모듈
│   ├── __init__.py
│   ├── README.md         # 유틸리티 모듈 설명
│   ├── config.py         # 설정 로더
│   ├── caching.py        # 캐싱 기능
│   └── reporting.py      # 보고서 생성 기능
│
├── assets/               # 정적 자원 (이미지 등)
│
└── data/                 # 데이터 디렉토리
    ├── input/            # 입력 파일
    ├── output/           # 출력 파일
    └── cache/            # 캐시 데이터