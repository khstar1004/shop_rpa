# 스크래핑 모듈

이 디렉토리는 외부 사이트에서 제품 정보를 수집하는 스크래퍼들을 포함합니다.

## 구성 요소

- `koryo_scraper.py`: 고려기프트 사이트에서 제품 정보를 검색하고 추출하는 스크래퍼
- `naver_crawler.py`: 네이버 쇼핑 웹사이트를 크롤링하여 제품 정보를 검색하고 추출하는 스크래퍼

## 고려기프트 스크래퍼 (koryo_scraper.py)

고려기프트 웹사이트에서 제품 정보를 검색하고 추출하는 기능:

- `KoryoScraper` 클래스: 고려기프트 사이트 접근 및 제품 검색
- `search_products()`: 키워드 기반으로 제품 검색
- `extract_product_details()`: 제품 상세 페이지에서 정보 추출
- `_parse_search_results()`: 검색 결과에서 제품 목록 파싱

## 네이버 쇼핑 크롤러 (naver_crawler.py)

네이버 쇼핑 웹사이트를 크롤링하여 제품 정보를 가져오는 기능:

- `NaverShoppingCrawler` 클래스: 네이버 쇼핑 페이지 크롤링
- `search_product()`: 키워드 기반으로 제품 검색
- `_crawl_page()`: 검색 결과 페이지 크롤링
- `_convert_to_product()`: 크롤링 결과를 `Product` 객체로 변환
- `is_promotional_site()`: 프로모션 사이트 여부 확인 로직

## 프로모션 사이트 식별

네이버 쇼핑 크롤러에서는 다음 키워드를 포함하는 상품 공급자를 프로모션 사이트로 식별합니다:

```python
PROMO_KEYWORDS = [
    "온오프마켓", "답례품", "기프트", "판촉", "기념품", 
    "인쇄", "각인", "제작", "홍보", "미스터몽키", "호갱탈출"
]
```

## 사용 예시

```python
# 고려기프트 스크래퍼 사용
from core.scraping.koryo_scraper import KoryoScraper

scraper = KoryoScraper()
products = scraper.search_product("텀블러")
if products:
    print(f"총 {len(products)}개의 제품을 찾았습니다.")

# 네이버 쇼핑 크롤러 사용
from core.scraping.naver_crawler import NaverShoppingCrawler

crawler = NaverShoppingCrawler()
products = crawler.search_product("텀블러")
if products:
    print(f"총 {len(products)}개의 제품을 찾았습니다.")
    print(f"프로모션 사이트 제품: {sum(p.is_promotional_site for p in products)}개")
``` 