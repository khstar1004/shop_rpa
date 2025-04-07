#!/usr/bin/env python3
"""
고려기프트 스크래퍼 테스트 실행 스크립트
모든 고려기프트 관련 테스트를 순차적으로 실행합니다.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# 테스트 스크립트 경로
TEST_SCRIPTS = [
    "tests/test_koryo_scraper.py",     # 기본 검색 테스트
    "tests/test_koryo_detailed.py"     # 상세 정보 추출 테스트
]

def run_tests(args):
    """모든 테스트 실행"""
    print(f"\n{'=' * 70}")
    print(f"        고려기프트 스크래퍼 테스트 시작 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'=' * 70}")
    
    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    
    # 캐시 클리어 옵션 처리
    if args.clear_cache:
        print("\n캐시 디렉토리 정리 중...")
        try:
            cache_files = [f for f in os.listdir("cache") if f.startswith("test_koryo")]
            for file in cache_files:
                os.remove(os.path.join("cache", file))
            print(f"캐시 파일 {len(cache_files)}개 삭제 완료")
        except Exception as e:
            print(f"캐시 정리 중 오류 발생: {str(e)}")
    
    # 각 테스트 스크립트 실행
    for i, script in enumerate(TEST_SCRIPTS, 1):
        if not os.path.exists(script):
            print(f"\n[오류] 테스트 스크립트를 찾을 수 없습니다: {script}")
            continue
            
        print(f"\n\n{'#' * 70}")
        print(f"# 테스트 {i}/{len(TEST_SCRIPTS)}: {os.path.basename(script)}")
        print(f"{'#' * 70}\n")
        
        try:
            # 스크립트 실행 (Python 경로를 명시적으로 지정)
            result = subprocess.run([sys.executable, script], check=False)
            
            if result.returncode == 0:
                print(f"\n✓ 테스트 성공: {os.path.basename(script)}")
            else:
                print(f"\n✗ 테스트 실패: {os.path.basename(script)} (종료 코드: {result.returncode})")
                
            # 상세 모드가 아니면 첫 번째 스크립트만 실행
            if not args.all and i == 1:
                print("\n간략 모드: 첫 번째 테스트만 실행했습니다.")
                print("모든 테스트를 실행하려면 --all 옵션을 사용하세요.")
                break
                
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
    
    print(f"\n{'=' * 70}")
    print(f"        고려기프트 스크래퍼 테스트 완료 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'=' * 70}\n")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="고려기프트 스크래퍼 테스트 실행")
    parser.add_argument("--all", action="store_true", help="모든 테스트 실행 (기본: 기본 테스트만)")
    parser.add_argument("--clear-cache", action="store_true", help="테스트 전 캐시 정리")
    args = parser.parse_args()
    
    run_tests(args)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 