#!/usr/bin/env python
"""
Excel 파일 처리 스크립트

PythonScript 폴더의 Excel 관련 기능을 활용하여 일반적인 Excel 작업을 수행합니다.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# 로깅 설정
def setup_logging(log_dir: Optional[str] = None) -> None:
    """로깅 설정"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 로그 디렉토리 설정
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"excel_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    logger = logging.getLogger()
    logger.info("로깅 설정 완료")
    return logger

def main():
    """메인 함수"""
    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description="Excel 파일 처리 유틸리티")
    parser.add_argument("--input-file", help="처리할 Excel 파일 경로")
    parser.add_argument("--input-dir", help="처리할 Excel 파일이 있는 디렉토리 (XLS 파일 변환에 사용)")
    parser.add_argument("--output-dir", help="결과 파일을 저장할 디렉토리")
    parser.add_argument("--log-dir", help="로그 파일을 저장할 디렉토리")
    
    # 기능 선택 옵션
    parser.add_argument("--convert-xls", action="store_true", help="XLS 파일을 XLSX로 변환")
    parser.add_argument("--check-columns", action="store_true", help="필요한 컬럼이 있는지 확인하고 추가")
    parser.add_argument("--add-hyperlinks", action="store_true", help="하이퍼링크 추가")
    parser.add_argument("--filter-price-diff", action="store_true", help="가격 차이가 있는 항목만 필터링")
    parser.add_argument("--all", action="store_true", help="모든 기능 적용 (위 옵션들의 조합)")
    
    # 이미지 삽입 관련 옵션
    parser.add_argument("--insert-image", action="store_true", help="이미지 삽입 모드")
    parser.add_argument("--image-path", help="삽입할 이미지 파일 경로")
    parser.add_argument("--target-cell", help="이미지를 삽입할 셀 주소 (예: A1)")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_dir)
    
    try:
        # utils 모듈에서 필요한 함수 가져오기
        from utils.excel_utils import (
            convert_xls_to_xlsx, 
            check_excel_columns, 
            add_hyperlinks_to_excel, 
            filter_excel_by_price_diff, 
            insert_image_to_excel,
            process_excel_file
        )
        
        # 이미지 삽입 모드
        if args.insert_image:
            if not args.image_path or not args.target_cell:
                logger.error("이미지 삽입 모드에서는 --image-path와 --target-cell을 모두 지정해야 합니다.")
                return 1
            
            result = insert_image_to_excel(args.image_path, args.target_cell)
            if result:
                logger.info("이미지가 성공적으로 삽입되었습니다.")
                return 0
            else:
                logger.error("이미지 삽입에 실패했습니다.")
                return 1
        
        # 처리할 파일이 지정되지 않은 경우 검사
        if not args.input_file and not args.input_dir:
            logger.error("--input-file 또는 --input-dir 중 하나가 필요합니다.")
            return 1
        
        # 전체 프로세스 실행
        if args.all:
            if args.input_file:
                result_path = process_excel_file(args.input_file)
                if result_path:
                    logger.info(f"파일 처리 완료: {result_path}")
                    return 0
                else:
                    logger.error("파일 처리에 실패했습니다.")
                    return 1
            else:
                logger.error("--all 옵션을 사용할 때는 --input-file이 필요합니다.")
                return 1
        
        # 개별 기능 실행
        result_path = None
        
        # XLS -> XLSX 변환
        if args.convert_xls:
            input_dir = args.input_dir or os.path.dirname(args.input_file)
            result_path = convert_xls_to_xlsx(input_dir)
            if result_path:
                logger.info(f"XLS 파일이 XLSX로 변환되었습니다: {result_path}")
                # 다음 단계에서 변환된 파일 사용
                args.input_file = result_path
            else:
                logger.warning("XLS 파일 변환에 실패했습니다.")
        
        # 컬럼 확인 및 추가
        if args.check_columns:
            if not args.input_file:
                logger.error("--check-columns 옵션을 사용할 때는 --input-file이 필요합니다.")
                return 1
            
            result = check_excel_columns(args.input_file)
            if result:
                logger.info(f"필요한 컬럼이 확인 및 추가되었습니다: {args.input_file}")
                result_path = args.input_file
            else:
                logger.error("컬럼 확인 및 추가에 실패했습니다.")
                return 1
        
        # 하이퍼링크 추가
        if args.add_hyperlinks:
            if not args.input_file:
                logger.error("--add-hyperlinks 옵션을 사용할 때는 --input-file이 필요합니다.")
                return 1
            
            input_file = result_path or args.input_file
            result_path = add_hyperlinks_to_excel(input_file)
            if result_path != input_file:
                logger.info(f"하이퍼링크가 추가된 파일이 생성되었습니다: {result_path}")
            else:
                logger.error("하이퍼링크 추가에 실패했습니다.")
                return 1
        
        # 가격 차이 필터링
        if args.filter_price_diff:
            if not result_path and not args.input_file:
                logger.error("--filter-price-diff 옵션을 사용할 때는 처리할 파일이 필요합니다.")
                return 1
            
            input_file = result_path or args.input_file
            result_path = filter_excel_by_price_diff(input_file)
            if result_path != input_file:
                logger.info(f"가격 차이가 필터링된 파일이 생성되었습니다: {result_path}")
            else:
                logger.warning("가격 차이 필터링에 실패했거나 필터링할 항목이 없습니다.")
        
        # 결과 출력
        if result_path:
            logger.info(f"모든 작업이 완료되었습니다. 최종 결과 파일: {result_path}")
            return 0
        else:
            logger.warning("작업은 완료되었지만 결과 파일이 생성되지 않았습니다.")
            return 1
        
    except Exception as e:
        logger.exception(f"처리 중 오류가 발생했습니다: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 