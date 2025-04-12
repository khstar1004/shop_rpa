"""
스크래핑 유틸리티 함수 모듈
"""

from bs4 import BeautifulSoup
from urllib.parse import urljoin

def extract_main_image(soup: BeautifulSoup, source: str, base_url: str = '') -> str:
    """
    웹사이트 소스에 따라 메인 이미지 URL을 추출합니다.
    
    Args:
        soup: BeautifulSoup 객체
        source: 웹사이트 소스 ('koryo' 또는 'haeoreum')
        base_url: 기본 URL (상대 경로를 절대 경로로 변환할 때 사용)
        
    Returns:
        str: 메인 이미지 URL
    """
    if source == 'koryo':
        # 고려기프트 메인 이미지 추출
        main_image = soup.select_one('.product .pic a img')
        if main_image:
            img_url = main_image.get('src', '')
            if img_url and not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(base_url, img_url)
            return img_url
            
    elif source == 'haeoreum':
        # 해오름 기프트 메인 이미지 추출
        # 1. target_img ID로 시도
        main_image = soup.select_one('img#target_img')
        if main_image:
            img_url = main_image.get('src', '')
            if img_url and not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(base_url, img_url)
            return img_url
            
        # 2. td[height="340"] img로 시도
        main_image = soup.select_one('td[height="340"] img')
        if main_image:
            img_url = main_image.get('src', '')
            if img_url and not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(base_url, img_url)
            return img_url
            
        # 3. 큰 이미지가 있는 모든 img 태그 검색
        large_images = soup.select('img[width="330"], img[height="330"], img[style*="max-height:330px"]')
        for img in large_images:
            img_url = img.get('src', '')
            if img_url and ('/upload/' in img_url or '/product/' in img_url):
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(base_url, img_url)
                return img_url
                
        # 4. 모든 이미지 태그 검색
        all_images = soup.find_all('img')
        for img in all_images:
            img_url = img.get('src', '')
            if img_url and ('/upload/' in img_url or '/product/' in img_url):
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(base_url, img_url)
                return img_url
                
    return '' 