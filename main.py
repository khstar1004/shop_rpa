import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.config import load_config
from utils.preprocessing import setup_logging
import logging

def main():
    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description="쇼핑 자동화 프로그램")
    parser.add_argument("--use_proxies", action="store_true", help="프록시 사용 여부")
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