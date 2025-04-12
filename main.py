import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.config import load_config
from utils.preprocessing import setup_logging
import logging
from core.processing.main_processor import ProductProcessor
from pathlib import Path

def get_absolute_path(relative_path: str) -> str:
    """Convert relative path to absolute path based on program root directory"""
    program_root = Path(__file__).parent.absolute()
    return str(program_root / relative_path)

def main():
    """메인 함수"""
    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description="쇼핑 자동화 프로그램")
    parser.add_argument("--use_proxies", action="store_true", help="프록시 사용 여부")
    parser.add_argument('--cli', action='store_true', help='CLI 모드 실행 (GUI 없이)')
    parser.add_argument('--input-files', nargs='+', help='CLI 모드에서 처리할 Excel 파일 목록')
    parser.add_argument('--output-dir', help='결과 파일을 저장할 디렉토리')
    parser.add_argument('--limit', type=int, help='처리할 상품 수 제한 (기본값: 전체 상품)')
    args = parser.parse_args()
    
    # Load configuration first to get paths
    try:
        config = load_config()
        
        # 명령줄 인자로 설정 값 업데이트
        if args.use_proxies:
            if 'NETWORK' not in config:
                config['NETWORK'] = {}
            config['NETWORK']['USE_PROXIES'] = 'True'
            
    except FileNotFoundError as e:
        print(f"오류: 설정 파일(config.ini)을 찾을 수 없습니다. {e}", file=sys.stderr)
        # Optionally create a default config here or exit
        sys.exit(1)
    except Exception as e:
        print(f"오류: 설정 파일 로딩 실패. {e}", file=sys.stderr)
        sys.exit(1)

    # Create necessary directories with absolute paths
    try:
        log_dir = get_absolute_path(config['PATHS']['LOG_DIR'])
        cache_dir = get_absolute_path(config['PATHS']['CACHE_DIR'])
        output_dir = get_absolute_path(config['PATHS']['OUTPUT_DIR'])
        intermediate_dir = get_absolute_path(config['PATHS']['INTERMEDIATE_DIR'])
        final_dir = get_absolute_path(config['PATHS']['FINAL_DIR'])
        
        # 디렉토리 생성 시도
        for dir_path in [log_dir, cache_dir, output_dir, intermediate_dir, final_dir]:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"디렉토리 생성/확인: {dir_path}")
            except OSError as e:
                print(f"경고: 디렉토리 생성 실패 ({dir_path}): {e}", file=sys.stderr)
                if dir_path == log_dir:
                    # 로그 디렉토리 생성 실패 시 대체 경로 사용
                    log_dir = os.path.join(os.path.expanduser("~"), "Shop_RPA_logs")
                    os.makedirs(log_dir, exist_ok=True)
                    print(f"대체 로그 디렉토리 사용: {log_dir}")
    except KeyError as e:
        print(f"오류: 설정 파일에 필요한 경로 키({e})가 없습니다.", file=sys.stderr)
        # 기본 로그 디렉토리 설정
        log_dir = os.path.join(os.path.expanduser("~"), "Shop_RPA_logs")
        os.makedirs(log_dir, exist_ok=True)
        print(f"기본 로그 디렉토리 사용: {log_dir}")
    
    # Setup logging (pass log_dir)
    try:
        setup_logging(log_dir=log_dir)
        logger = logging.getLogger(__name__)
        logger.info("애플리케이션 시작")
        logger.info(f"로그 디렉토리: {log_dir}")
    except Exception as e:
        print(f"로깅 설정 실패: {e}", file=sys.stderr)
        # 기본 로깅 설정으로 폴백
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error(f"로깅 설정 실패, 기본 로깅 사용: {e}")
    
    # 프록시 사용 여부 로깅
    if args.use_proxies:
        logger.info("프록시 사용 모드로 실행됩니다.")
    
    # CLI 모드로 실행
    if args.cli and args.input_files:
        try:
            # 프로세서 초기화
            processor = ProductProcessor(config)
            
            # 파일 처리
            output_files = processor.process_files(args.input_files, args.output_dir, args.limit)
            
            if output_files:
                print(f"처리 완료: {len(output_files)}개 파일이 생성되었습니다.")
                for file in output_files:
                    print(f"- {file}")
            else:
                print("처리 실패: 출력 파일이 생성되지 않았습니다.")
                
            sys.exit(0)
        except Exception as e:
            logger.error(f"CLI 모드 처리 중 오류 발생: {str(e)}", exc_info=True)
            print(f"오류 발생: {str(e)}")
            sys.exit(1)
    else:
        try:
            # Initialize and run GUI application
            app = QApplication(sys.argv)
            # Pass the loaded config to MainWindow
            window = MainWindow(config)
            window.show()
            logger.info("메인 윈도우 표시됨")
            sys.exit(app.exec_())
        except Exception as e:
            logger.critical(f"애플리케이션 실행 중 심각한 오류 발생: {e}", exc_info=True)
            sys.exit(1)

if __name__ == '__main__':
    main() 