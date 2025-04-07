# 네이버 쇼핑 크롤러 (Naver Shopping Crawler)

이 문서는 네이버 쇼핑 API를 이용한 상품 검색 도구 설정 및 사용법을 설명합니다.

## 개요

`NaverShoppingCrawler` 클래스는 네이버 쇼핑 API를 이용해 상품을 검색하고 결과를 구조화된 데이터로 변환합니다. 이 클래스는 다음 기능을 제공합니다:

- 네이버 쇼핑 API를 이용한 상품 검색
- 가격 기반 필터링 (참조 가격 대비 10% 이상 차이나는 상품만 표시)
- 결과 캐싱 (성능 향상 및 API 호출 제한 준수)
- 비동기 처리 (여러 페이지 동시 검색)
- 결과 없음 처리 (동일상품 없음 메시지 생성)

## 설정 방법

### 1. 네이버 개발자 API 키 발급

1. [네이버 개발자 센터](https://developers.naver.com)에 가입 및 로그인
2. 애플리케이션 등록 (Application > 애플리케이션 등록)
3. 애플리케이션 이름 설정 (예: "Shop RPA")
4. API 선택: 검색 > 쇼핑
5. 환경 추가: WEB 설정, 서비스 URL 입력 (로컬 테스트 시 http://localhost)
6. 애플리케이션 등록 완료 후 Client ID와 Client Secret 저장

### 2. .env 파일 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가합니다:

```
client_id = "발급받은_Client_ID"
client_secret = "발급받은_Client_Secret"
```

### 3. 패키지 설치

필요한 패키지가 모두 설치되어 있는지 확인합니다:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 사용법

```python
from core.scraping.naver_crawler import NaverShoppingCrawler
from utils.caching import FileCache

# 캐시 초기화 (선택 사항)
cache = FileCache(cache_dir="cache", max_size_mb=100)

# 크롤러 초기화
crawler = NaverShoppingCrawler(
    max_retries=3,
    cache=cache,
    timeout=30
)

# 제품 검색
products = crawler.search_product("텀블러", max_items=10)

# 검색 결과 출력
for product in products:
    print(f"상품명: {product.name}")
    print(f"가격: {product.price}원")
    print(f"판매처: {product.brand}")
    print(f"URL: {product.url}")
    print(f"이미지: {product.image_url}")
    print("---")
```

### 참조 가격을 이용한 검색

```python
# 참조 가격 설정 (10% 룰 적용)
reference_price = 10000
products = crawler.search_product("텀블러", reference_price=reference_price)

# 검색 결과는 참조 가격보다 낮거나, 10% 이상 높은 제품만 포함됩니다
```

## 문제 해결

### API 키 오류

API 요청 시 다음과 같은 오류가 발생할 경우:

```
{"errorMessage":"Scope Status Invalid : Authentication failed. (인증에 실패했습니다.)","errorCode":"024"}
```

다음을 확인하세요:

1. `.env` 파일이 프로젝트 루트 디렉토리에 있는지 확인
2. Client ID와 Client Secret이 올바르게 입력되었는지 확인
3. 네이버 개발자 센터에서 API 사용 권한이 제대로 설정되었는지 확인
4. API 호출 한도를 초과하지 않았는지 확인

### 캐싱 문제

캐싱 관련 오류가 발생할 경우:

1. `cache` 디렉토리가 존재하는지 확인
2. 캐시 디렉토리 쓰기 권한이 있는지 확인
3. 다음 명령어로 캐시를 초기화할 수 있습니다:

```python
cache.clear()  # 캐시 전체 삭제
```

## 고급 설정

### 설정 파일 (config.ini)

`config.ini`의 `SCRAPING` 섹션에서 네이버 크롤러의 동작을 세부적으로 조정할 수 있습니다:

```ini
[SCRAPING]
MAX_PAGES = 3                    # 최대 검색 페이지 수
REQUIRE_IMAGE_MATCH = true       # 이미지 매칭 필수 여부
POLITENESS_DELAY = 2000          # API 요청 간 지연 시간 (밀리초)
```

## 라이선스

이 프로젝트는 네이버 API 이용 약관을 준수해야 합니다. 자세한 내용은 [네이버 개발자 API 이용약관](https://developers.naver.com/products/terms/)을 참고하세요. 