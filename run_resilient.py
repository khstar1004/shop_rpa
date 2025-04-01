#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
네이버 쇼핑 크롤링 차단 우회 및 복원력 있는 실행 스크립트
- 418 오류(봇 감지)나 기타 오류로 인해 프로그램이 중단되면 자동으로 재시작
- 로깅 기능으로 문제 추적
- 지수 백오프 알고리즘으로 재시도 간격 설정
"""

import os
import sys
import time
import subprocess
import random
import logging
import argparse
from datetime import datetime

# 로깅 설정
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"resilient_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("resilient_runner")

def run_with_resilience(max_retries=10, use_proxies=False):
    """
    애플리케이션을 실행하고, 문제 발생 시 자동으로 재시작
    
    Args:
        max_retries: 최대 재시도 횟수
        use_proxies: 프록시 사용 여부
    """
    logger = setup_logging()
    retry_count = 0
    base_delay = 5  # 기본 대기 시간(초)
    
    # 명령줄 인자 준비 
    cmd = [sys.executable, "main.py"]
    if use_proxies:
        cmd.append("--use_proxies")
        
    logger.info(f"복원력 있는 실행 시작: {' '.join(cmd)}")
    
    while retry_count < max_retries:
        try:
            # 프로세스 시작
            logger.info(f"실행 시도 #{retry_count+1}/{max_retries}")
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # 표준 출력 및 오류 로깅
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"APP: {output.strip()}")
                    
                # 418 에러나 크롤링 관련 오류 감지
                if "HTTP 418" in output or "Error fetching page" in output:
                    logger.warning("크롤링 차단 감지됨")
            
            # 표준 오류 로깅
            for error in process.stderr.readlines():
                logger.error(f"APP ERROR: {error.strip()}")
                
            # 종료 코드 확인
            return_code = process.wait()
            
            # 정상 종료인 경우 (0)
            if return_code == 0:
                logger.info("프로그램이 정상 종료되었습니다.")
                break
                
            # 비정상 종료인 경우
            logger.error(f"프로그램이 비정상 종료되었습니다. 반환 코드: {return_code}")
            retry_count += 1
            
            # 지수 백오프 적용
            delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(1, 5)
            logger.info(f"{delay:.2f}초 후 재시작합니다...")
            time.sleep(delay)
            
        except KeyboardInterrupt:
            logger.info("사용자가 프로그램을 중단했습니다.")
            break
        except Exception as e:
            logger.error(f"실행 중 오류 발생: {str(e)}")
            retry_count += 1
            delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(1, 5)
            logger.info(f"{delay:.2f}초 후 재시작합니다...")
            time.sleep(delay)
    
    if retry_count >= max_retries:
        logger.error(f"최대 재시도 횟수({max_retries})에 도달했습니다. 프로그램을 종료합니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="복원력 있는 쇼핑 자동화 실행기")
    parser.add_argument("--max_retries", type=int, default=10, help="최대 재시도 횟수 (기본값: 10)")
    parser.add_argument("--use_proxies", action="store_true", help="프록시 사용 여부")
    
    args = parser.parse_args()
    run_with_resilience(max_retries=args.max_retries, use_proxies=args.use_proxies) 