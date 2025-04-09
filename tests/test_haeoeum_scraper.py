#!/usr/bin/env python
"""
Haeoeum Gift 스크래퍼 테스트 스크립트

이 스크립트는 해오름 기프트(JCL Gift) 웹사이트에서 상품 정보와 이미지를 추출하는
HaeoeumScraper의 기능을 테스트합니다.
"""

import logging
import sys
import os
from pprint import pprint

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# core 패키지 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 스크래퍼 임포트
from core.scraping.haeoeum_scraper import HaeoeumScraper

def test_product_extraction(product_idx: str):
    """특정 상품 추출 테스트"""
    scraper = HaeoeumScraper(debug=True)
    logger = logging.getLogger("test")
    
    logger.info(f"Extracting product with ID: {product_idx}")
    product = scraper.get_product(product_idx)
    
    if product:
        logger.info(f"Successfully extracted product: {product.name}")
        logger.info(f"Product code: {product.product_code}")
        logger.info(f"Main image URL: {product.image_url}")
        
        if product.image_gallery:
            logger.info(f"Found {len(product.image_gallery)} images in total")
            for i, img_url in enumerate(product.image_gallery):
                logger.info(f"Image {i+1}: {img_url}")
        else:
            logger.warning("No images found in the gallery")
    else:
        logger.error(f"Failed to extract product with ID: {product_idx}")

def main():
    """메인 함수"""
    # 테스트할 상품 ID 목록
    product_ids = [
        "431692",
        "431677",
        "418619",
        "418617",
        "418616",
        "418615"
    ]
    
    for product_idx in product_ids:
        print(f"\n{'='*50}")
        print(f"Testing product ID: {product_idx}")
        print(f"{'='*50}")
        test_product_extraction(product_idx)
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main() 