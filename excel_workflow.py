#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Excel 가격 처리 워크플로우 실행 스크립트

1차 파일 처리 → 2차 파일 생성 워크플로우를 실행하는 간단한 명령행 스크립트입니다.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from utils.excel_processor import run_complete_workflow

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('excel_workflow.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='Excel 가격 처리 워크플로우 실행 스크립트')
    
    parser.add_argument('input_file', help='처리할 Excel 파일 경로')
    parser.add_argument('--output-dir', '-o', help='출력 디렉토리 (기본값: 입력 파일과 같은 위치)', default=None)
    parser.add_argument('--skip-first-stage', action='store_true', help='1차 처리 건너뛰기 (이미 존재하는 1차 파일 사용)')
    parser.add_argument('--skip-second-stage', action='store_true', help='2차 처리 건너뛰기 (1차 파일만 생성)')
    parser.add_argument('--email', action='store_true', help='결과 이메일 전송')
    parser.add_argument('--email-recipient', default='dasomas@kakao.com', help='이메일 수신자 (기본값: dasomas@kakao.com)')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"워크플로우 시작: {args.input_file}")
    
    try:
        # 파일 존재 확인
        if not os.path.exists(args.input_file):
            logger.error(f"입력 파일을 찾을 수 없음: {args.input_file}")
            return 1
            
        # 출력 디렉토리 설정
        if not args.output_dir:
            args.output_dir = os.path.dirname(args.input_file)
            
        # 워크플로우 실행
        results = run_complete_workflow(
            input_file=args.input_file,
            output_root_dir=args.output_dir,
            run_first_stage=not args.skip_first_stage,
            run_second_stage=not args.skip_second_stage,
            send_email=args.email
        )
        
        # 결과 출력
        print("\n=== 처리 결과 ===")
        print(f"상태: {results['status']}")
        
        if results.get('first_stage_file'):
            print(f"1차 파일: {results['first_stage_file']}")
            
        if results.get('second_stage_file'):
            print(f"2차 파일: {results['second_stage_file']}")
            
        print(f"소요 시간: {results.get('duration_seconds', 0):.2f}초")
        
        if args.email:
            print(f"이메일 전송: {'성공' if results.get('email_sent') else '실패'}")
        
        # 오류 처리
        if results['status'] == 'error':
            print(f"오류: {results.get('error', '알 수 없는 오류')}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"워크플로우 실행 중 오류 발생: {str(e)}", exc_info=True)
        print(f"\n오류 발생: {str(e)}")
        return 1
        
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 