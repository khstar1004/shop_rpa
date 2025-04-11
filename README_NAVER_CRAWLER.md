# 네이버 쇼핑 크롤러 구현 상세

## 개요

네이버 쇼핑 크롤러는 웹 크롤링을 통해 상품 정보를 수집하는 모듈입니다. API 대신 웹 크롤링을 사용하여 더 유연하고 안정적인 데이터 수집이 가능합니다.

## 주요 기능

### 1. 웹 크롤링
- Selenium을 사용한 동적 웹 페이지 크롤링
- 자동 로그인 및 세션 관리
- IP 차단 방지를 위한 딜레이 설정
- User-Agent 랜덤화

### 2. 데이터 수집
- 상품명, 가격, 판매처 정보 수집
- 프로모션 정보 자동 감지
- 상품 이미지 URL 수집
- 상품 상세 정보 수집

### 3. 필터링
- 키워드 기반 필터링
- 가격 범위 필터링
- 판매처 필터링
- 프로모션 상품 필터링

### 4. 캐싱
- 파일 기반 캐싱 시스템
- 캐시 만료 시간 설정
- 캐시 크기 제한
- 캐시 무효화 기능

## 사용 방법

### 1. 초기화
```python
from core.scraping.naver_crawler import NaverShoppingCrawler
from utils.caching import FileCache

# 캐시 설정
cache = FileCache(
    cache_dir="cache",
    max_size_mb=100,
    expiration_hours=24
)

# 크롤러 초기화
crawler = NaverShoppingCrawler(
    max_retries=3,
    cache=cache,
    timeout=30,
    delay=2
)
```

### 2. 상품 검색
```python
# 기본 검색
products = crawler.search_product("검색어", max_items=10)

# 필터링 옵션 사용
products = crawler.search_product(
    "검색어",
    max_items=10,
    min_price=10000,
    max_price=50000,
    exclude_keywords=["중고", "리퍼"],
    only_promotion=True
)
```

### 3. 상세 정보 수집
```python
# 상품 상세 정보 수집
product_details = crawler.get_product_details(product_id)
```

## 설정 옵션

### 1. 크롤러 설정
- `max_retries`: 최대 재시도 횟수 (기본값: 3)
- `timeout`: 요청 타임아웃 (초) (기본값: 30)
- `delay`: 요청 간 딜레이 (초) (기본값: 2)
- `user_agent`: User-Agent 문자열 (기본값: 랜덤)

### 2. 캐시 설정
- `cache_dir`: 캐시 디렉토리 경로
- `max_size_mb`: 최대 캐시 크기 (MB)
- `expiration_hours`: 캐시 만료 시간 (시간)

## 문제 해결

### 1. 크롤링 실패
- 네트워크 연결 확인
- IP 차단 여부 확인
- 딜레이 시간 조정
- User-Agent 변경

### 2. 데이터 누락
- 필터링 조건 확인
- 페이지 로딩 대기 시간 조정
- 재시도 횟수 증가

### 3. 성능 문제
- 캐시 크기 조정
- 배치 크기 조정
- 동시 요청 수 제한

## 주의사항

1. 네트워크 사용량
   - 과도한 요청은 IP 차단의 원인이 될 수 있습니다
   - 적절한 딜레이를 설정하여 사용하세요

2. 데이터 정확성
   - 웹 페이지 구조 변경 시 크롤러 업데이트 필요
   - 정기적인 테스트 수행 권장

3. 법적 고려사항
   - robots.txt 준수
   - 서비스 약관 준수
   - 개인정보 보호 준수 