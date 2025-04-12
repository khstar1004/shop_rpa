from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re

def test_image_extraction():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # 테스트할 URL
        url = 'https://adpanchok.co.kr/ez/mall.php?cat=013001000&query=view&no=122793'
        
        print(f"\n테스트 URL: {url}")
        page.goto(url, wait_until='networkidle')
        # 이미지 로딩을 위해 잠시 대기
        time.sleep(2)
        
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        # 상품 이미지 찾기
        product_images = []
        
        # 1. 메인 이미지
        main_img = soup.find('img', id='main_img')
        if main_img and main_img.get('src'):
            main_img_url = urljoin(url, main_img['src'])
            if main_img_url not in product_images:
                product_images.append(main_img_url)
        
        # 2. 썸네일 이미지들
        thumbnails = soup.select('.product_picture .thumnails img')
        for thumb in thumbnails:
            if thumb.get('src'):
                thumb_url = urljoin(url, thumb['src'])
                if thumb_url not in product_images:
                    product_images.append(thumb_url)
        
        # 3. 상세 이미지들 (상품 정보 테이블 내 이미지)
        detail_images = soup.select('.tbl_info img')
        for img in detail_images:
            if img.get('src'):
                img_url = img['src']
                # 상대 경로인 경우 절대 경로로 변환
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(url, img_url)
                if img_url not in product_images:
                    product_images.append(img_url)
        
        # 4. 상세 이미지들 (상품 설명 영역)
        # 모든 테이블에서 이미지 찾기
        tables = soup.find_all('table')
        for table in tables:
            images = table.find_all('img')
            for img in images:
                if img.get('src'):
                    img_url = img['src']
                    # 상대 경로인 경우 절대 경로로 변환
                    if not img_url.startswith(('http://', 'https://')):
                        img_url = urljoin(url, img_url)
                    if img_url not in product_images:
                        product_images.append(img_url)
        
        # 5. 리뷰 이미지들
        review_images = soup.select('.tbl_review img')
        for img in review_images:
            if img.get('src'):
                img_url = urljoin(url, img['src'])
                if img_url not in product_images:
                    product_images.append(img_url)
        
        # 6. 외부 도메인 이미지들 (상세 설명용)
        external_images = soup.find_all('img', src=re.compile(r'ain8949\.godohosting\.com'))
        for img in external_images:
            if img.get('src'):
                img_url = img['src']
                if img_url not in product_images:
                    product_images.append(img_url)
        
        if product_images:
            print('상품 이미지 URL들:')
            for i, url in enumerate(product_images, 1):
                print(f"{i}. {url}")
        else:
            print('상품 이미지를 찾을 수 없습니다.')
        
        browser.close()

if __name__ == '__main__':
    test_image_extraction() 