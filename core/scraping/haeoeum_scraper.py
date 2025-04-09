"""
Haeoeum Gift 웹사이트 스크래퍼 모듈

Haeoeum Gift(JCL Gift) 웹사이트에서 상품 정보를 추출하는 기능을 제공합니다.
특히 product_view.asp 페이지에서 이미지 및 상품 정보를 추출하는 기능에 최적화되어 있습니다.
"""

import logging
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import time
import hashlib
from pathlib import Path

from . import BaseMultiLayerScraper
from ..data_models import Product

class HaeoeumScraper(BaseMultiLayerScraper):
    """
    해오름 기프트(JCL Gift) 웹사이트 스크래퍼
    특히 이미지 URL이 제대로 추출되도록 최적화되어 있습니다.
    """
    
    BASE_URL = "http://www.jclgift.com"
    PRODUCT_VIEW_URL = "http://www.jclgift.com/product/product_view.asp"
    
    def __init__(self, 
                 timeout: int = 10, 
                 max_retries: int = 3,
                 cache: Optional[Any] = None,
                 debug: bool = False):
        """스크래퍼 초기화"""
        super().__init__(max_retries=max_retries, timeout=timeout, cache=cache)
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        
        # 해오름 기프트 웹사이트 셀렉터 정의
        self.selectors = {
            # 상품명 관련 셀렉터
            'product_title': {
                'selector': 'td.Ltext[bgcolor="#F5F5F5"]',
                'options': {'multiple': False}
            },
            # 상품 코드 관련 셀렉터
            'product_code': {
                'selector': 'td.code_b2 > button > b',
                'options': {'multiple': False}
            },
            # 메인 이미지 관련 셀렉터 (target_img)
            'main_image': {
                'selector': 'img#target_img',
                'options': {'multiple': False, 'attribute': 'src'}
            },
            # 대체 메인 이미지 셀렉터
            'alt_main_image': {
                'selector': 'td[height="340"] img',
                'options': {'multiple': False, 'attribute': 'src'}
            },
            # 썸네일 이미지 셀렉터
            'thumbnail_images': {
                'selector': 'table[width="62"] img',
                'options': {'multiple': True, 'attribute': 'src'}
            },
            # 가격 관련 셀렉터들
            'applied_price': {
                'selector': '#price_e',
                'options': {'multiple': False}
            },
            'total_price': {
                'selector': '#buy_price',
                'options': {'multiple': False}
            },
            # 가격표 셀렉터
            'price_table': {
                'selector': 'table.pv_tbl',
                'options': {'multiple': False}
            },
            # 상품 상세 정보 테이블
            'product_info_table': {
                'selector': 'table.tbl_contab',
                'options': {'multiple': False}
            },
            # 상품 설명 이미지
            'description_images': {
                'selector': 'div.product_view_img img',
                'options': {'multiple': True, 'attribute': 'src'}
            }
        }
        
        # 텍스트 추출용 정규식 패턴
        self.patterns = {
            'price_number': re.compile(r'[\d,]+'),
            'product_code': re.compile(r'상품코드\s*:\s*([A-Za-z0-9-]+)')
        }
    
    def get_product(self, product_idx: str) -> Optional[Product]:
        """
        해오름 기프트 상품 페이지에서 상품 정보를 추출합니다.
        
        Args:
            product_idx: 상품 ID (p_idx 파라미터 값)
            
        Returns:
            Product 객체 또는 None (실패 시)
        """
        # 캐시 확인
        cache_key = f"haeoeum_product|{product_idx}"
        cached_result = self.get_sparse_data(cache_key)
        if cached_result:
            self.logger.info(f"Using cached product data for p_idx={product_idx}")
            return cached_result
        
        url = f"{self.PRODUCT_VIEW_URL}?p_idx={product_idx}"
        self.logger.info(f"Fetching product data from: {url}")
        
        try:
            # 웹 페이지 요청
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    break
                except (requests.RequestException, Exception) as e:
                    self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    if attempt + 1 == self.max_retries:
                        raise
                    time.sleep(1)
            
            # HTML 파싱
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 상품 정보 추출
            product_data = self._extract_product_data(soup, product_idx, url)
            
            if product_data:
                # Product 객체 생성
                product = Product(
                    id=product_data.get('product_id', ''),
                    name=product_data.get('title', ''),
                    source='haeoreum',
                    price=float(product_data.get('price', 0)),
                    url=url,
                    image_url=product_data.get('main_image', ''),
                    product_code=product_data.get('product_code', ''),
                    image_gallery=product_data.get('image_gallery', [])
                )
                
                # 캐시에 저장
                self.add_sparse_data(cache_key, product, ttl=86400)  # 24시간 캐싱
                return product
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching product data: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None
    
    def _extract_product_data(self, soup: BeautifulSoup, product_idx: str, url: str) -> Dict[str, Any]:
        """
        BeautifulSoup 객체에서 상품 정보를 추출합니다.
        """
        product_data = {}
        
        # 상품명 추출
        title_element = soup.select_one(self.selectors['product_title']['selector'])
        title = title_element.text.strip() if title_element else ""
        
        if not title:
            # 타이틀 태그에서 추출 시도
            title = soup.title.string.strip() if soup.title else ""
            if '>' in title:
                # 타이틀에서 불필요한 사이트명 제거
                title = title.split('>', 1)[0].strip()
        
        product_data['title'] = title
        
        # 상품 코드 추출
        code_element = soup.select_one(self.selectors['product_code']['selector'])
        product_code = code_element.text.strip() if code_element else ""
        
        # 코드가 없으면 URL에서 추출 시도
        if not product_code:
            product_code = product_idx
        
        product_data['product_code'] = product_code
        
        # 고유 ID 생성
        product_id = hashlib.md5(f"haeoeum_{product_code}".encode()).hexdigest()
        product_data['product_id'] = product_id
        
        # 메인 이미지 URL 추출 - 중요: ID가 target_img인 이미지 찾기
        main_image_element = soup.select_one(self.selectors['main_image']['selector'])
        if main_image_element:
            main_image = main_image_element.get('src', '')
        else:
            # 대체 셀렉터 시도
            alt_image_element = soup.select_one(self.selectors['alt_main_image']['selector'])
            main_image = alt_image_element.get('src', '') if alt_image_element else ""
        
        # URL 정규화
        if main_image and not main_image.startswith(('http://', 'https://')):
            main_image = urljoin(self.BASE_URL, main_image)
        
        product_data['main_image'] = main_image
        
        # 모든 이미지 URL 추출 (메인 + 썸네일 + 설명 이미지)
        image_gallery = []
        
        # 메인 이미지 추가
        if main_image:
            image_gallery.append(main_image)
        
        # 썸네일 이미지 추가
        thumbnail_elements = soup.select(self.selectors['thumbnail_images']['selector'])
        for img in thumbnail_elements:
            img_url = img.get('src', '')
            if img_url and not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(self.BASE_URL, img_url)
            
            # 중복 제거하고 추가
            if img_url and img_url not in image_gallery:
                image_gallery.append(img_url)
        
        # 설명 이미지 추가
        desc_img_elements = soup.select(self.selectors['description_images']['selector'])
        for img in desc_img_elements:
            img_url = img.get('src', '')
            if img_url and not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(self.BASE_URL, img_url)
            
            # 중복 제거하고 추가
            if img_url and img_url not in image_gallery:
                image_gallery.append(img_url)
        
        # 페이지 내 모든 img 태그 검색하여 놓친 이미지 없는지 확인
        all_images = soup.find_all('img')
        for img in all_images:
            img_url = img.get('src', '')
            # 필터링: 실제 상품 이미지만 추가, 아이콘 등은 제외
            if img_url and ('/upload/' in img_url or '/product/' in img_url):
                if img_url and not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(self.BASE_URL, img_url)
                # 중복 제거하고 추가
                if img_url and img_url not in image_gallery:
                    image_gallery.append(img_url)
        
        product_data['image_gallery'] = image_gallery
        
        # 가격 추출
        price_element = soup.select_one(self.selectors['applied_price']['selector'])
        price_text = price_element.text.strip() if price_element else "0"
        price_match = self.patterns['price_number'].search(price_text)
        price = int(price_match.group().replace(',', '')) if price_match else 0
        
        # 가격이 0이면 총합계금액에서 시도
        if price == 0:
            total_price_element = soup.select_one(self.selectors['total_price']['selector'])
            if total_price_element:
                price_text = total_price_element.text.strip()
                price_match = self.patterns['price_number'].search(price_text)
                price = int(price_match.group().replace(',', '')) if price_match else 0
        
        product_data['price'] = price
        
        # 수량별 가격 추출
        quantity_prices = {}
        price_table = soup.select_one(self.selectors['price_table']['selector'])
        if price_table:
            rows = price_table.select('tr')
            if len(rows) >= 2:
                quantity_cells = rows[0].select('td')[1:]  # 첫번째 셀(수량)은 제외
                price_cells = rows[1].select('td')[1:]    # 첫번째 셀(단가)은 제외
                
                for i in range(min(len(quantity_cells), len(price_cells))):
                    qty_text = quantity_cells[i].text.strip()
                    price_text = price_cells[i].text.strip()
                    
                    # 숫자만 추출
                    qty_match = re.search(r'\d+', qty_text)
                    price_match = self.patterns['price_number'].search(price_text)
                    
                    if qty_match and price_match:
                        qty = int(qty_match.group())
                        qty_price = int(price_match.group().replace(',', ''))
                        quantity_prices[str(qty)] = qty_price
        
        product_data['quantity_prices'] = quantity_prices
        
        # 사양 정보 추출
        specifications = {}
        specs_table = soup.select_one(self.selectors['product_info_table']['selector'])
        if specs_table:
            rows = specs_table.select('tr')
            for row in rows:
                th = row.select_one('th')
                td = row.select_one('td')
                if th and td:
                    key = th.text.strip().replace('·', '').strip()
                    value = td.text.strip()
                    if key and value:
                        specifications[key] = value
        
        product_data['specifications'] = specifications
        
        return product_data 