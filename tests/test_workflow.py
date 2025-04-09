#!/usr/bin/env python
"""
해오름 기프트 워크플로우 테스트

이 스크립트는 해오름 기프트 상품 정보를 가져오고, 텍스트/이미지를 대조하여 
고려기프트 및 네이버와 매칭한 후 엑셀에 결과를 작성하는 전체 워크플로우를 검증합니다.
"""

import logging
import sys
import os
import time
import pandas as pd
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 패키지 경로 추가 - 상위 디렉토리를 추가하여 utils 모듈을 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 구성 및 필요한 클래스 가져오기
from utils.config import load_config
from core.processing.main_processor import ProductProcessor
from core.scraping.haeoeum_scraper import HaeoeumScraper

def create_test_data_file(product_ids, output_file='test_input.xlsx'):
    """테스트용 엑셀 파일 생성"""
    logger = logging.getLogger("test")
    logger.info(f"Creating test input file with {len(product_ids)} products")
    
    data = []
    
    # 해오름 스크래퍼를 사용하여 실제 제품 정보 가져오기
    scraper = HaeoeumScraper(debug=True)
    
    for idx, p_idx in enumerate(product_ids):
        try:
            # 스크래퍼를 통해 제품 정보 가져오기
            product = scraper.get_product(p_idx)
            
            if product:
                # 기본 데이터
                row_data = {
                    '구분': 'A',
                    '담당자': '테스트',
                    '업체명': 'JCL기프트',
                    '업체코드': product.product_code,
                    '상품Code': product.product_code,
                    '중분류카테고리': '테스트',
                    '상품명': product.name,
                    '기본수량(1)': 100,
                    '판매단가(V포함)': product.price,
                    '본사상품링크': f"http://www.jclgift.com/product/product_view.asp?p_idx={p_idx}",
                    '본사 이미지': product.image_url
                }
                
                data.append(row_data)
                logger.info(f"Added product: {product.name}")
            else:
                logger.warning(f"Failed to fetch product data for ID: {p_idx}")
        except Exception as e:
            logger.error(f"Error processing product ID {p_idx}: {str(e)}")
    
    # 데이터프레임 생성 및 파일 저장
    if data:
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        logger.info(f"Test file created: {output_file} with {len(data)} products")
        return output_file
    else:
        logger.error("No data to write to Excel")
        return None

def test_full_workflow():
    """전체 워크플로우 테스트 - 해오름 → 텍스트/이미지 매칭 → 엑셀 작성"""
    logger = logging.getLogger("test")
    logger.info("Starting full workflow test")
    
    # 설정 로드
    config = load_config()
    
    # 테스트할 상품 ID 목록 (해오름 기프트 p_idx)
    product_ids = [
        "431692",  # 첫 번째 상품
        "431677",  # 두 번째 상품
        "418619",  # 세 번째 상품
        "418617",  # 네 번째 상품
        "418616",  # 다섯 번째 상품
        "418615"   # 여섯 번째 상품
    ]
    
    # 테스트 입력 파일 생성
    test_file = create_test_data_file(product_ids)
    if not test_file:
        logger.error("Failed to create test input file")
        return False
    
    # 프로세서 초기화
    processor = ProductProcessor(config)
    
    # 파일 처리
    logger.info(f"Processing test file: {test_file}")
    start_time = time.time()
    output_file, error = processor.process_file(test_file)
    end_time = time.time()
    
    if error:
        logger.error(f"Error processing file: {error}")
        return False
    
    if output_file:
        logger.info(f"Successfully processed file in {end_time - start_time:.2f} seconds")
        logger.info(f"Output file: {output_file}")
        
        # 결과 파일 확인
        if os.path.exists(output_file):
            result_df = pd.read_excel(output_file)
            logger.info(f"Result file contains {len(result_df)} rows")
            
            # 매칭 성공 여부 확인
            logger.info("\n===== 매칭 결과 요약 =====")
            
            # 고려기프트 매칭 확인
            koryo_matched = sum(1 for x in result_df['고려기프트 상품링크'] if x and str(x).strip())
            logger.info(f"고려기프트 매칭 성공: {koryo_matched}/{len(result_df)} 제품")
            
            # 네이버 매칭 확인
            naver_matched = sum(1 for x in result_df['네이버 쇼핑 링크'] if x and str(x).strip())
            logger.info(f"네이버 매칭 성공: {naver_matched}/{len(result_df)} 제품")
            
            # 해오름 이미지 확인
            haeoeum_images = sum(1 for x in result_df['본사 이미지'] if x and str(x).strip())
            logger.info(f"해오름 이미지 있음: {haeoeum_images}/{len(result_df)} 제품")
            
            # 이미지 테스트
            if haeoeum_images < len(result_df):
                logger.warning("일부 제품에 이미지가 없습니다!")
            else:
                logger.info("모든 제품에 이미지가 있습니다 ✓")
            
            # 테스트 정리
            logger.info("\n===== 워크플로우 테스트 완료 =====")
            logger.info(f"총 처리 시간: {end_time - start_time:.2f}초")
            logger.info(f"결과 파일: {output_file}")
            return True
        else:
            logger.error(f"Output file not found: {output_file}")
            return False
    else:
        logger.error("No output file produced")
        return False

if __name__ == "__main__":
    test_full_workflow() 