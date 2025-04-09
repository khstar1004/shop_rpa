#!/usr/bin/env python
"""
Haeoeum Gift 스크래퍼 이미지 테스트

이 스크립트는 해오름 기프트 스크래퍼가 제대로 상품명과 이미지를 가져오는지 테스트합니다.
"""

import logging
import sys
import os
from pprint import pprint
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 패키지 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 스크래퍼 임포트
from core.scraping.haeoeum_scraper import HaeoeumScraper

def test_product_images(product_idx):
    """특정 상품의 이미지 추출 테스트"""
    logger = logging.getLogger("test")
    scraper = HaeoeumScraper(debug=True)
    
    logger.info(f"Extracting product with ID: {product_idx}")
    start_time = time.time()
    product = scraper.get_product(product_idx)
    end_time = time.time()
    
    if product:
        print(f"\n{'='*80}")
        print(f"Product: {product.name}")
        print(f"Product Code: {product.product_code}")
        print(f"Price: {product.price:,} 원")
        print(f"{'='*80}")
        
        # 메인 이미지 정보
        print(f"\n[메인 이미지]")
        print(f"{product.image_url}")
        
        # 모든 이미지 정보
        if product.image_gallery:
            print(f"\n[전체 이미지 갤러리 ({len(product.image_gallery)}개)]")
            for i, img_url in enumerate(product.image_gallery):
                print(f"{i+1}. {img_url}")
        else:
            print("\n[이미지 없음]")
        
        # 추출 시간 정보
        print(f"\n[처리 시간: {end_time - start_time:.2f}초]")
        print(f"{'='*80}\n")
        return True
    else:
        logger.error(f"Failed to extract product with ID: {product_idx}")
        return False

def main():
    """메인 함수"""
    # 테스트할 상품 ID 목록
    product_ids = [
        "431692",  # 첫 번째 상품
        "431677",  # 두 번째 상품
        "418619",  # 세 번째 상품
        "418617",  # 네 번째 상품
        "418616",  # 다섯 번째 상품
        "418615"   # 여섯 번째 상품
    ]
    
    success_count = 0
    fail_count = 0
    
    for product_idx in product_ids:
        print(f"\nTesting product ID: {product_idx}")
        if test_product_images(product_idx):
            success_count += 1
        else:
            fail_count += 1
    
    # 결과 요약
    print(f"\n{'='*80}")
    print(f"테스트 결과: 총 {len(product_ids)}개 상품 중 {success_count}개 성공, {fail_count}개 실패")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 