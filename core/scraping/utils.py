"""
스크래핑 유틸리티 함수 모듈
"""

import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict, Any, Optional

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
    # 사이트별 메인 이미지 선택자 정의
    selectors = {
        'koryo': ['.product .pic a img', '.goods-view-image img', '.goods_image img'],
        'haeoreum': [
            'img#target_img', 
            'img[style*="cursor:hand"][onclick*="view_big"]',
            'img[width="330"][height="330"]',
            'td[height="340"] img', 
            'img[width="330"]', 
            'img[height="330"]'
        ]
    }
    
    # 이미지 경로 패턴 정의
    valid_paths = ['/upload/', '/product/', '/data/item/', '/images/']
    
    # 사이트별 선택자 가져오기
    site_selectors = selectors.get(source, [])
    
    # 선택자로 이미지 찾기
    for selector in site_selectors:
        img = soup.select_one(selector)
        if img and img.get('src'):
            img_url = img.get('src')
            # 상대 경로를 절대 경로로 변환
            if img_url and not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(base_url, img_url)
                
            # 해오름 기프트의 경우 onclick 속성에서 큰 이미지 추출 시도
            if source == 'haeoreum' and img.get('onclick') and 'view_big' in img.get('onclick'):
                import re
                big_img_match = re.search(r"view_big\('([^']+)'", img.get('onclick'))
                if big_img_match:
                    big_img_url = big_img_match.group(1)
                    if not big_img_url.startswith(('http://', 'https://')):
                        big_img_url = urljoin(base_url, big_img_url)
                    return big_img_url
                
            return img_url
    
    # 선택자로 찾지 못한 경우 경로 패턴으로 검색
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if not src:
            continue
            
        # 유효한 이미지 경로인지 확인
        if any(path in src for path in valid_paths):
            # 상대 경로를 절대 경로로 변환
            if not src.startswith(('http://', 'https://')):
                src = urljoin(base_url, src)
            return src
    
    # 이미지를 찾지 못한 경우 빈 문자열 반환
    return ''

def extract_images(soup: BeautifulSoup, base_url: str = '') -> List[str]:
    """
    HTML에서 모든 유효한 이미지 URL을 추출합니다.
    
    Args:
        soup: BeautifulSoup 객체
        base_url: 기본 URL (상대 경로를 절대 경로로 변환할 때 사용)
        
    Returns:
        List[str]: 이미지 URL 목록
    """
    images = set()  # 중복 방지
    
    # 이미지 태그에서 src 속성 추출
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if src and _is_valid_image_url(src):
            # 상대 경로를 절대 경로로 변환
            if not src.startswith(('http://', 'https://')):
                src = urljoin(base_url, src)
            images.add(src)
    
    # 백그라운드 이미지 추출
    for element in soup.find_all(style=True):
        style = element.get('style', '')
        if 'url(' in style:
            urls = re.findall(r'url\([\'"]?([^\'"()]+)[\'"]?\)', style)
            for url in urls:
                if _is_valid_image_url(url):
                    # 상대 경로를 절대 경로로 변환
                    if not url.startswith(('http://', 'https://')):
                        url = urljoin(base_url, url)
                    images.add(url)
    
    return list(images)

def _is_valid_image_url(url: str) -> bool:
    """
    URL이 유효한 이미지 URL인지 확인합니다.
    
    Args:
        url: 확인할 URL
        
    Returns:
        bool: 유효한 이미지 URL이면 True, 아니면 False
    """
    # 이미지 확장자 확인
    if re.search(r'\.(jpe?g|png|gif|webp|bmp)([\?#].*)?$', url.lower()):
        # 제외할 패턴 (아이콘, 버튼 등)
        exclude = ['icon', 'button', 'btn', 'bg_', 'pixel.gif', 'blank.gif', 'spacer.gif']
        return not any(pattern in url.lower() for pattern in exclude)
    return False

def normalize_url(url: str, base_url: str = '') -> str:
    """
    URL을 정규화합니다.
    
    Args:
        url: 정규화할 URL
        base_url: 기본 URL (상대 경로를 절대 경로로 변환할 때 사용)
        
    Returns:
        str: 정규화된 URL
    """
    if not url:
        return ''
        
    # 상대 경로를 절대 경로로 변환
    if not url.startswith(('http://', 'https://')):
        url = urljoin(base_url, url)
        
    # 쿼리 파라미터 제거 (필요한 경우)
    url = url.split('#')[0]  # 프래그먼트 제거
    
    return url 