import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.config import load_config
from utils.preprocessing import setup_logging
import logging
from core.processing.main_processor import ProductProcessor

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

    # Create necessary directories
    try:
        log_dir = config['PATHS']['LOG_DIR']
        os.makedirs(config['PATHS']['CACHE_DIR'], exist_ok=True)
        os.makedirs(config['PATHS']['OUTPUT_DIR'], exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"오류: 필수 디렉토리 생성 실패 ({e}). 권한을 확인하세요.", file=sys.stderr)
        # Log dir might fail, try logging to console only
        log_dir = None 
    except KeyError as e:
        print(f"오류: 설정 파일에 필요한 경로 키({e})가 없습니다.", file=sys.stderr)
        log_dir = None # Fallback
    
    # Setup logging (pass log_dir)
    setup_logging(log_dir=log_dir)
    logger = logging.getLogger(__name__)
    logger.info("애플리케이션 시작")
    
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
            # Pass the loaded config to MainWindow along with the limit argument
            window = MainWindow(config, product_limit=args.limit)
            window.show()
            logger.info("메인 윈도우 표시됨")
            sys.exit(app.exec_())
        except Exception as e:
            logger.critical(f"애플리케이션 실행 중 심각한 오류 발생: {e}", exc_info=True)
            sys.exit(1)

if __name__ == '__main__':
    main() 