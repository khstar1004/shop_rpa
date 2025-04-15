"""
고려기프트 스크래핑 결과가 엑셀에 올바르게 반영되는지 테스트하는 스크립트
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_koryo_excel")

# 프로젝트 루트 디렉토리에 대한 상대 경로 설정
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# 유틸리티 임포트
from utils.excel_processor import run_scraping_for_excel
from core.scraping.koryo_scraper import KoryoScraper
from core.scraping.haeoeum_scraper import HaeoeumScraper
from core.scraping.naver_crawler import NaverShoppingAPI


def create_test_df():
    """테스트를 위한 데이터프레임 생성"""
    # 1. 정상 상품 (찾을 수 있을만한 상품)
    # 2. 검색 결과가 없을 상품 (동일상품 없음)
    # 3. 상품명 정보가 없는 경우
    
    data = {
        '상품명': [
            '3단우산', 
            '이상한이름의상품12345',
            ''
        ],
        '고려기프트 상품링크': [
            'https://adpanchok.co.kr/ez/mall.php?query=view&no=1234',
            'https://adpanchok.co.kr/ez/mall.php?query=view&no=9999',
            'https://adpanchok.co.kr/ez/mall.php?query=view&no=5678'
        ],
        # 기본 이미지 URL 추가 - 빈 값으로 초기화
        '고려기프트 이미지': [
            '',
            '',
            ''
        ],
        # 기본 상태 추가 - 빈 값으로 초기화
        '고려기프트 상태': [
            '',
            '',
            ''
        ]
    }
    
    return pd.DataFrame(data)


def run_test():
    """테스트 실행"""
    logger.info("고려기프트 엑셀 스크래핑 테스트 시작")
    
    # 테스트 데이터 생성
    df = create_test_df()
    logger.info(f"테스트 데이터:\n{df}")
    
    # 스크래핑 실행
    result_df = run_scraping_for_excel(df)
    
    # 결과 확인
    logger.info("스크래핑 결과:")
    display_cols = ['상품명', '고려기프트 이미지', '고려기프트 상태', '고려_가격']
    available_cols = [col for col in display_cols if col in result_df.columns]
    
    logger.info(f"사용 가능한 컬럼: {available_cols}")
    logger.info(f"모든 컬럼: {list(result_df.columns)}")
    
    if set(display_cols).issubset(set(result_df.columns)):
        logger.info(f"결과:\n{result_df[display_cols]}")
    else:
        missing_cols = set(display_cols) - set(result_df.columns)
        logger.warning(f"일부 컬럼이 없습니다: {missing_cols}")
        logger.info(f"가능한 결과:\n{result_df}")
    
    # 결과 검증
    success = True
    
    # 상태 메시지 컬럼 확인
    if '고려기프트 상태' in result_df.columns:
        # 상태값 전체 출력
        logger.info("고려기프트 상태 값 전체:")
        for idx, status in enumerate(result_df['고려기프트 상태']):
            logger.info(f"  행 {idx}: {status}")
            
        # 두 번째 행은 '동일상품 없음' 상태여야 함
        expected_status = '동일상품 없음'
        actual_status = result_df.loc[1, '고려기프트 상태']
        logger.info(f"검증: 행 1 기대 상태: '{expected_status}', 실제 상태: '{actual_status}'")
        if actual_status != expected_status:
            logger.error(f"검색 결과가 없는 상품의 상태가 '{expected_status}'이 아닙니다: '{actual_status}'")
            success = False
            
        # 세 번째 행은 '상품명 정보 없음' 상태여야 함  
        expected_status = '상품명 정보 없음'
        actual_status = result_df.loc[2, '고려기프트 상태']
        logger.info(f"검증: 행 2 기대 상태: '{expected_status}', 실제 상태: '{actual_status}'")
        if actual_status != expected_status:
            logger.error(f"상품명이 없는 상품의 상태가 '{expected_status}'이 아닙니다: '{actual_status}'")
            success = False
    else:
        logger.error("'고려기프트 상태' 컬럼이 결과에 없습니다")
        success = False
        
    # 이미지 URL 컬럼 확인  
    if '고려기프트 이미지' in result_df.columns:
        logger.info("고려기프트 이미지 값 전체:")
        for idx, img_url in enumerate(result_df['고려기프트 이미지']):
            logger.info(f"  행 {idx}: {img_url}")
            
        for i in range(3):  # 모든 행 검사
            if not pd.notna(result_df.loc[i, '고려기프트 이미지']) or not result_df.loc[i, '고려기프트 이미지']:
                logger.error(f"{i+1}번 행의 이미지 URL이 비어 있습니다")
                success = False
    else:
        logger.error("'고려기프트 이미지' 컬럼이 결과에 없습니다")
        success = False
        
    if success:
        logger.info("테스트 성공: 고려기프트 스크래핑 결과가 엑셀에 올바르게 반영됨")
    else:
        logger.error("테스트 실패: 고려기프트 스크래핑 결과가 엑셀에 올바르게 반영되지 않음")
        
    return result_df


def test_koryo_direct():
    """직접 고려기프트 스크래핑 테스트 (프레임워크 없이)"""
    logger.info("고려기프트 직접 테스트 시작")
    
    # 초기화
    from utils.caching import FileCache
    cache = FileCache(ttl=3600, max_size_mb=1024, compression=True)
    koryo_scraper = KoryoScraper(cache=cache)
    
    # 테스트 케이스 정의
    test_cases = [
        {'name': '3단우산', 'expected_status': '동일상품 없음'},
        {'name': '이상한이름의상품12345', 'expected_status': '동일상품 없음'},
        {'name': '', 'expected_status': '상품명 정보 없음'}
    ]
    
    results = []
    
    # 각 케이스 실행
    for test in test_cases:
        product_name = test['name']
        expected_status = test['expected_status']
        
        try:
            # 상품명이 있는 경우 검색 시도
            if product_name:
                # async 함수 실행 (헬퍼 함수 사용)
                from utils.excel_processor import run_async_func
                products = run_async_func(koryo_scraper.search_product, product_name, max_items=1)
                
                if products and len(products) > 0:
                    product = products[0]
                    result_status = product.status if hasattr(product, 'status') else "상태 없음"
                    result_image = product.image_url if hasattr(product, 'image_url') else "이미지 없음"
                else:
                    result_status = "검색 결과 없음 (빈 리스트)"
                    result_image = "이미지 없음"
            else:
                # 상품명이 없는 경우
                result_status = "상품명 정보 없음"
                result_image = f"{koryo_scraper.base_url}/img/no_image.jpg"
            
            # 결과 검증
            status_match = result_status == expected_status
            logger.info(f"테스트 - 상품명: '{product_name}', 상태: '{result_status}' (기대: '{expected_status}'), 일치: {status_match}")
            logger.info(f"  이미지: {result_image}")
            
            results.append({
                '상품명': product_name,
                '상태': result_status,
                '기대 상태': expected_status,
                '일치 여부': status_match,
                '이미지': result_image
            })
            
        except Exception as e:
            logger.error(f"직접 테스트 오류 (상품: {product_name}): {e}")
            results.append({
                '상품명': product_name,
                '상태': f"오류: {e}",
                '기대 상태': expected_status,
                '일치 여부': False,
                '이미지': '오류로 인해 확인 불가'
            })
    
    # 종합 결과
    success = all(r['일치 여부'] for r in results)
    status = "성공" if success else "실패"
    logger.info(f"직접 테스트 완료: {status}")
    
    return {
        '결과': status,
        '테스트 케이스': len(results),
        '성공': sum(1 for r in results if r['일치 여부']),
        '세부 결과': results
    }


if __name__ == "__main__":
    # 기존 테스트 실행
    result = run_test()
    logger.info("테스트 완료")
    
    # 직접 테스트도 실행
    direct_result = test_koryo_direct()
    logger.info(f"직접 테스트 결과: {direct_result}") 