#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
from datetime import datetime
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 추가 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 라이브러리 로깅 레벨 조정
logging.getLogger('playwright').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# 현재 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.caching import FileCache
from utils.config import load_config
from core.processing.main_processor import ProductProcessor
from core.data_models import Product

# 가격 불량 판정 기준 (%)
PRICE_DIFF_THRESHOLD = 10.0

def main():
    """워크플로우 테스트 메인 함수"""
    logging.info("=== 가격조사 워크플로우 테스트 시작 ===")
    
    # 설정 로드
    try:
        config = load_config()
        logging.info("설정 파일 로드 완료")
    except Exception as e:
        logging.error(f"설정 파일 로드 실패: {str(e)}")
        return

    # 캐시 초기화
    cache = FileCache(
        cache_dir=config["PATHS"]["CACHE_DIR"],
        duration_seconds=86400,  # 1일
        max_size_mb=1024,  # 1GB
    )
    
    # 출력 디렉토리 생성
    os.makedirs(config["PATHS"]["OUTPUT_DIR"], exist_ok=True)
    
    # ProductProcessor 초기화
    processor = ProductProcessor(config)
    logging.info("ProductProcessor 초기화 완료")
    
    # 테스트용 해오름 기프트 상품 생성 (가격 정보 중심)
    test_products = [
        Product(
            id="test_product_1",
            name="크리스탈 트로피",
            source="haeoreum",
            price=50000,
            url="http://www.jclgift.com/product/product_view.asp?p_idx=11245",
            image_url="http://www.jclgift.com/pdata/goods/s/21/s1021236.jpg",
            brand="해오름",
            stock_status="판매중",  # 재고 상태 추가
            original_input_data={
                "본사 이미지": "http://www.jclgift.com/pdata/goods/s/21/s1021236.jpg",
                "본사상품링크": "http://www.jclgift.com/product/product_view.asp?p_idx=11245",
                "상품명": "크리스탈 트로피",
                "규격": "10cm x 5cm x 20cm",
                "판매가": "50000",
                "재고상태": "판매중",
                "구분": "A"  # A: 승인관리, P: 가격관리
            }
        ),
        Product(
            id="test_product_2",
            name="유리 머그컵",
            source="haeoreum",
            price=15000,
            url="http://www.jclgift.com/product/product_view.asp?p_idx=10578",
            image_url="http://www.jclgift.com/pdata/goods/s/20/s1020578.jpg",
            brand="해오름",
            stock_status="판매중",
            original_input_data={
                "본사 이미지": "http://www.jclgift.com/pdata/goods/s/20/s1020578.jpg",
                "본사상품링크": "http://www.jclgift.com/product/product_view.asp?p_idx=10578",
                "상품명": "유리 머그컵",
                "규격": "지름 8cm x 높이 10cm",
                "판매가": "15000",
                "재고상태": "판매중",
                "구분": "A"
            }
        ),
        Product(
            id="test_product_3",
            name="볼펜 세트",
            source="haeoreum",
            price=10000,
            url="http://www.jclgift.com/product/product_view.asp?p_idx=5869",
            image_url="http://www.jclgift.com/pdata/goods/s/14/s1015869.jpg",
            brand="해오름",
            stock_status="품절",  # 품절 상태 테스트
            original_input_data={
                "본사 이미지": "http://www.jclgift.com/pdata/goods/s/14/s1015869.jpg",
                "본사상품링크": "http://www.jclgift.com/product/product_view.asp?p_idx=5869",
                "상품명": "볼펜 세트",
                "규격": "13.5cm",
                "판매가": "10000",
                "재고상태": "품절",
                "구분": "P"  # 가격관리
            }
        )
    ]
    
    # 타임스탬프로 고유한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config["PATHS"]["OUTPUT_DIR"], f"가격조사_{timestamp}.xlsx")
    
    # 각 상품에 대해 워크플로우 실행
    results = []
    price_issue_products = []  # 가격 불량 상품 목록
    stock_issue_products = []  # 재고 불량 상품 목록
    
    for product in test_products:
        logging.info(f"\n{'='*30}")
        logging.info(f"가격조사 시작: {product.name}")
        logging.info(f"해오름 상품 정보:")
        logging.info(f"  - 상품ID: {product.id}")
        logging.info(f"  - 상품명: {product.name}")
        logging.info(f"  - 판매가: {product.price}원")
        logging.info(f"  - 재고상태: {product.stock_status}")
        logging.info(f"  - 구분: {product.original_input_data.get('구분', 'N/A')}")
        logging.info(f"{'='*30}")
        
        # 상품 처리
        result = processor.process_product(product)
        results.append(result)
        
        # 결과 로그
        logging.info(f"\n{'='*30}")
        logging.info(f"가격조사 완료: {product.name}")
        
        # 고려기프트 매칭 결과 및 가격 비교
        has_price_issue = False
        has_stock_issue = False
        
        if result.best_koryo_match:
            matched = result.best_koryo_match.matched_product
            logging.info(f"\n>>> 고려기프트 매칭 결과 <<<")
            logging.info(f"  상품명: {matched.name}")
            logging.info(f"  판매가: {matched.price}원")
            
            # 가격 차이 계산 및 출력
            price_diff = matched.price - product.price
            price_diff_percent = (price_diff / product.price) * 100 if product.price > 0 else 0
            
            if abs(price_diff_percent) > PRICE_DIFF_THRESHOLD:
                has_price_issue = True
                status_msg = "⚠️ 가격 불량"
            else:
                status_msg = "✅ 가격 정상"
                
            logging.info(f"  가격 차이: {price_diff}원 ({price_diff_percent:.1f}%) {status_msg}")
            
            # 재고 상태 확인 (if available)
            if hasattr(matched, 'stock_status') and matched.stock_status:
                match_stock = matched.stock_status
                if product.stock_status != match_stock:
                    has_stock_issue = True
                    logging.info(f"  재고 상태: {match_stock} ⚠️ 재고 불일치")
                else:
                    logging.info(f"  재고 상태: {match_stock} ✅ 재고 일치")
        else:
            logging.info("\n❌ 고려기프트 매칭 없음")
        
        # 네이버 매칭 결과
        if result.best_naver_match:
            matched = result.best_naver_match.matched_product
            logging.info(f"\n>>> 네이버 매칭 결과 <<<")
            logging.info(f"  상품명: {matched.name}")
            logging.info(f"  판매가: {matched.price}원")
            logging.info(f"  판매처: {matched.brand or getattr(matched, 'mall_name', 'N/A')}")
            
            # 가격 차이 계산 및 출력
            price_diff = matched.price - product.price
            price_diff_percent = (price_diff / product.price) * 100 if product.price > 0 else 0
            
            if abs(price_diff_percent) > PRICE_DIFF_THRESHOLD:
                has_price_issue = True
                status_msg = "⚠️ 가격 불량"
            else:
                status_msg = "✅ 가격 정상"
                
            logging.info(f"  가격 차이: {price_diff}원 ({price_diff_percent:.1f}%) {status_msg}")
            
            # 재고 상태 확인 (if available)
            if hasattr(matched, 'stock_status') and matched.stock_status:
                match_stock = matched.stock_status
                if product.stock_status != match_stock:
                    has_stock_issue = True
                    logging.info(f"  재고 상태: {match_stock} ⚠️ 재고 불일치")
                else:
                    logging.info(f"  재고 상태: {match_stock} ✅ 재고 일치")
        else:
            logging.info("\n❌ 네이버 매칭 없음")
        
        # 가격/재고 불량 상품 목록에 추가
        if has_price_issue:
            price_issue_products.append((product, result))
            logging.info(f"\n⚠️ 가격 불량 상품: {product.name}")
        
        if has_stock_issue:
            stock_issue_products.append((product, result))
            logging.info(f"\n⚠️ 재고 불량 상품: {product.name}")
        
        logging.info(f"{'='*30}\n")
            
    # 엑셀 파일로 저장 (전체 결과)
    try:
        # 결과를 데이터프레임으로 변환
        logging.info("엑셀 파일 생성 중...")
        
        # 비교 결과 데이터 준비
        comparison_data = []
        koryo_comparison_data = []  # 고려기프트 매칭 데이터
        naver_comparison_data = []  # 네이버 매칭 데이터
        
        for result in results:
            source = result.source_product
            koryo_match = result.best_koryo_match.matched_product if result.best_koryo_match else None
            naver_match = result.best_naver_match.matched_product if result.best_naver_match else None
            
            # 가격 불량 여부 계산
            price_issue = False
            stock_issue = False
            koryo_price_issue = False
            naver_price_issue = False
            
            # 원본 데이터
            original_data = source.original_input_data.copy() if source.original_input_data else {}
            
            # 기본 해오름 데이터
            row = {
                "구분": original_data.get("구분", ""),
                "상품ID": source.id,
                "상품명": source.name,
                "판매가": source.price,
                "재고상태": source.stock_status,
                "상태": "정상",  # 기본값
                "매칭완료": "X",  # 기본값
                "본사URL": source.url,
                "본사이미지URL": source.image_url,
                "규격": original_data.get("규격", ""),
            }
            
            # 고려기프트 매칭 결과 분석
            if koryo_match:
                row["매칭완료"] = "O"
                
                koryo_price_diff = koryo_match.price - source.price
                koryo_price_diff_percent = (koryo_price_diff / source.price) * 100 if source.price > 0 else 0
                koryo_price_issue = abs(koryo_price_diff_percent) > PRICE_DIFF_THRESHOLD
                
                # 재고 불일치 확인
                koryo_stock_issue = False
                if hasattr(koryo_match, 'stock_status') and koryo_match.stock_status != source.stock_status:
                    koryo_stock_issue = True
                    stock_issue = True
                
                # 고려기프트 매칭 데이터 추가
                koryo_row = {
                    "구분": original_data.get("구분", ""),
                    "상품ID": source.id,
                    "원본상품명": source.name,
                    "매칭상품명": koryo_match.name,
                    "본사판매가": source.price,
                    "고려판매가": koryo_match.price,
                    "가격차이": koryo_price_diff,
                    "가격차이(%)": f"{koryo_price_diff_percent:.1f}%",
                    "가격상태": "불량" if koryo_price_issue else "정상",
                    "본사재고": source.stock_status,
                    "고려재고": getattr(koryo_match, 'stock_status', ""),
                    "재고상태": "불일치" if koryo_stock_issue else "일치",
                    "매칭유사도": f"{result.best_koryo_match.combined_similarity:.2f}",
                    "본사URL": source.url,
                    "고려URL": koryo_match.url,
                    "판매처": "고려기프트",
                }
                koryo_comparison_data.append(koryo_row)
                
                if koryo_price_issue:
                    price_issue = True
            
            # 네이버 매칭 결과 분석
            if naver_match:
                row["매칭완료"] = "O"
                
                naver_price_diff = naver_match.price - source.price
                naver_price_diff_percent = (naver_price_diff / source.price) * 100 if source.price > 0 else 0
                naver_price_issue = abs(naver_price_diff_percent) > PRICE_DIFF_THRESHOLD
                
                # 재고 불일치 확인
                naver_stock_issue = False
                if hasattr(naver_match, 'stock_status') and naver_match.stock_status != source.stock_status:
                    naver_stock_issue = True
                    stock_issue = True
                
                # 네이버 매칭 데이터 추가
                naver_row = {
                    "구분": original_data.get("구분", ""),
                    "상품ID": source.id,
                    "원본상품명": source.name,
                    "매칭상품명": naver_match.name,
                    "본사판매가": source.price,
                    "네이버판매가": naver_match.price,
                    "가격차이": naver_price_diff,
                    "가격차이(%)": f"{naver_price_diff_percent:.1f}%",
                    "가격상태": "불량" if naver_price_issue else "정상",
                    "본사재고": source.stock_status,
                    "네이버재고": getattr(naver_match, 'stock_status', ""),
                    "재고상태": "불일치" if naver_stock_issue else "일치",
                    "매칭유사도": f"{result.best_naver_match.combined_similarity:.2f}",
                    "본사URL": source.url,
                    "네이버URL": naver_match.url,
                    "판매처": naver_match.brand or getattr(naver_match, 'mall_name', 'N/A'),
                }
                naver_comparison_data.append(naver_row)
                
                if naver_price_issue:
                    price_issue = True
            
            # 상태 업데이트
            if price_issue:
                row["상태"] = "가격 불량"
            if stock_issue:
                row["상태"] = "재고 불량" if not price_issue else "가격+재고 불량"
            
            comparison_data.append(row)
        
        # DataFrames 생성
        df_main = pd.DataFrame(comparison_data)
        df_koryo = pd.DataFrame(koryo_comparison_data) if koryo_comparison_data else None
        df_naver = pd.DataFrame(naver_comparison_data) if naver_comparison_data else None
        
        # 엑셀 작성자 설정 (with 여러 시트)
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name="전체 상품", index=False)
            if df_koryo is not None:
                df_koryo.to_excel(writer, sheet_name="고려기프트 매칭", index=False)
            if df_naver is not None:
                df_naver.to_excel(writer, sheet_name="네이버 매칭", index=False)
        
        logging.info(f"결과 저장 완료: {output_file}")
        
        # 엑셀 후처리 (이미지 포뮬러 추가 등)
        try:
            processed_file = processor.excel_manager.post_process_excel_file(output_file)
            logging.info(f"결과 파일 후처리 완료: {processed_file}")
        except Exception as e:
            logging.error(f"결과 파일 후처리 중 오류: {str(e)}")
        
        # 가격 불량 상품만 별도 파일로 저장
        if price_issue_products:
            price_issue_file = output_file.replace('.xlsx', '_가격불량.xlsx')
            
            # 가격 불량 상품 데이터 추출
            price_issue_data = []
            for source, result in price_issue_products:
                koryo_match = result.best_koryo_match.matched_product if result.best_koryo_match else None
                naver_match = result.best_naver_match.matched_product if result.best_naver_match else None
                
                row = {
                    "구분": source.original_input_data.get("구분", ""),
                    "상품ID": source.id,
                    "상품명": source.name,
                    "본사판매가": source.price,
                    "본사URL": source.url,
                    "본사이미지URL": source.image_url,
                    "규격": source.original_input_data.get("규격", ""),
                }
                
                if koryo_match:
                    price_diff_percent = ((koryo_match.price - source.price) / source.price * 100) if source.price > 0 else 0
                    price_diff_status = "불량" if abs(price_diff_percent) > PRICE_DIFF_THRESHOLD else "정상"
                    
                    row.update({
                        "고려기프트판매가": koryo_match.price,
                        "고려기프트가격차이": koryo_match.price - source.price,
                        "고려기프트가격차이(%)": f"{price_diff_percent:.1f}%",
                        "고려기프트가격상태": price_diff_status,
                        "고려기프트URL": koryo_match.url,
                        "고려기프트매칭유사도": f"{result.best_koryo_match.combined_similarity:.2f}",
                    })
                
                if naver_match:
                    price_diff_percent = ((naver_match.price - source.price) / source.price * 100) if source.price > 0 else 0
                    price_diff_status = "불량" if abs(price_diff_percent) > PRICE_DIFF_THRESHOLD else "정상"
                    
                    row.update({
                        "네이버판매가": naver_match.price,
                        "네이버가격차이": naver_match.price - source.price,
                        "네이버가격차이(%)": f"{price_diff_percent:.1f}%",
                        "네이버가격상태": price_diff_status,
                        "네이버판매처": naver_match.brand or getattr(naver_match, 'mall_name', 'N/A'),
                        "네이버URL": naver_match.url,
                        "네이버매칭유사도": f"{result.best_naver_match.combined_similarity:.2f}",
                    })
                
                price_issue_data.append(row)
            
            # 가격 불량 상품 저장
            df_price_issue = pd.DataFrame(price_issue_data)
            df_price_issue.to_excel(price_issue_file, index=False, sheet_name="가격불량상품")
            
            logging.info(f"가격 불량 상품 {len(price_issue_products)}개 저장 완료: {price_issue_file}")
            
            # 엑셀 후처리 (이미지 포뮬러 추가 등)
            try:
                processed_file = processor.excel_manager.post_process_excel_file(price_issue_file)
                logging.info(f"가격 불량 파일 후처리 완료: {processed_file}")
            except Exception as e:
                logging.error(f"가격 불량 파일 후처리 중 오류: {str(e)}")
        else:
            logging.info("가격 불량 상품이 없습니다.")
        
        # 재고 불량 상품만 별도 파일로 저장
        if stock_issue_products:
            stock_issue_file = output_file.replace('.xlsx', '_재고불량.xlsx')
            
            # 재고 불량 상품 데이터 추출
            stock_issue_data = []
            for source, result in stock_issue_products:
                koryo_match = result.best_koryo_match.matched_product if result.best_koryo_match else None
                naver_match = result.best_naver_match.matched_product if result.best_naver_match else None
                
                row = {
                    "구분": source.original_input_data.get("구분", ""),
                    "상품ID": source.id,
                    "상품명": source.name,
                    "본사재고상태": source.stock_status,
                    "본사URL": source.url,
                }
                
                if koryo_match and hasattr(koryo_match, 'stock_status'):
                    row.update({
                        "고려기프트재고상태": koryo_match.stock_status,
                        "고려기프트재고일치": "X" if koryo_match.stock_status != source.stock_status else "O",
                        "고려기프트URL": koryo_match.url,
                    })
                
                if naver_match and hasattr(naver_match, 'stock_status'):
                    row.update({
                        "네이버재고상태": naver_match.stock_status,
                        "네이버재고일치": "X" if naver_match.stock_status != source.stock_status else "O",
                        "네이버판매처": naver_match.brand or getattr(naver_match, 'mall_name', 'N/A'),
                        "네이버URL": naver_match.url,
                    })
                
                stock_issue_data.append(row)
            
            # 재고 불량 상품 저장
            df_stock_issue = pd.DataFrame(stock_issue_data)
            df_stock_issue.to_excel(stock_issue_file, index=False, sheet_name="재고불량상품")
            
            logging.info(f"재고 불량 상품 {len(stock_issue_products)}개 저장 완료: {stock_issue_file}")
        else:
            logging.info("재고 불량 상품이 없습니다.")
        
        # 결과 통계 출력
        logging.info("\n=== 가격조사 결과 통계 ===")
        logging.info(f"총 상품 수: {len(test_products)}개")
        logging.info(f"가격 불량 상품 수: {len(price_issue_products)}개")
        logging.info(f"재고 불량 상품 수: {len(stock_issue_products)}개")
        logging.info(f"매칭 실패 상품 수: {sum(1 for r in results if not r.best_koryo_match and not r.best_naver_match)}개")
            
    except Exception as e:
        logging.error(f"결과 저장 중 오류 발생: {str(e)}", exc_info=True)
    
    logging.info("=== 가격조사 워크플로우 테스트 완료 ===")

if __name__ == "__main__":
    main() 