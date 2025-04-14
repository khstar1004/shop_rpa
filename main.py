import sys
import os
import argparse
import importlib
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.config import load_config
from utils.preprocessing import setup_logging
import logging
from core.processing.main_processor import ProductProcessor
from pathlib import Path
import pandas as pd

# 작업메뉴얼 워크플로우 사용법 문서
MANUAL_WORKFLOW_HELP = """
작업메뉴얼 기준 워크플로우 실행 방법:

1. 상품명 전처리, 파일 분할, 네이버·고려기프트 검색, 1차/2차 파일 처리를 모두 실행:
   python main.py --cli --input-files data/example.xlsx --manual-workflow --split

2. 이메일 전송 비활성화:
   python main.py --cli --input-files data/example.xlsx --manual-workflow --no-email

3. 2차 파일 생성 비활성화:
   python main.py --cli --input-files data/example.xlsx --manual-workflow --no-second-stage

4. 특정 이메일 주소로 보내기:
   python main.py --cli --input-files data/example.xlsx --manual-workflow --email-recipient someone@example.com

5. 특정 출력 디렉토리 지정:
   python main.py --cli --input-files data/example.xlsx --manual-workflow --output-dir /path/to/output

주의사항:
- 구분(A/P) 값은 입력파일과 출력파일 간에 유지됩니다.
- 이메일 전송을 위해서는 환경변수(EMAIL_USER, EMAIL_PASSWORD, SMTP_SERVER)를 설정해야 합니다.
- 기본 이메일 수신자는 dasomas@kakao.com입니다.
"""

def get_absolute_path(relative_path: str) -> str:
    """Convert relative path to absolute path based on program root directory"""
    program_root = Path(__file__).parent.absolute()
    return str(program_root / relative_path)

def split_large_file(file_path: str, max_items: int = 300) -> list:
    """
    작업메뉴얼 기준 - 대용량 파일 분할 처리
    한 번에 300개 이상 처리 불가능하여 여러 파일로 분할
    
    Args:
        file_path: 입력 파일 경로
        max_items: 파일당 최대 항목 수 (기본값: 300)
    
    Returns:
        생성된 분할 파일 경로 리스트
    """
    try:
        logging.info(f"대용량 파일 분할 처리 시작: {file_path}")
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            logging.error(f"파일이 존재하지 않습니다: {file_path}")
            return [file_path]  # 오류 시 원본 경로 반환
            
        # 파일 확장자 확인
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in ['.xls', '.xlsx', '.xlsm']:
            logging.error(f"지원되지 않는 파일 형식입니다: {ext}")
            return [file_path]  # 오류 시 원본 경로 반환
        
        # 파일 로드
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logging.error(f"엑셀 파일 로드 중 오류 발생: {str(e)}")
            return [file_path]  # 오류 시 원본 경로 반환
            
        total_rows = len(df)
        
        # 분할 필요 없는 경우
        if total_rows <= max_items:
            logging.info(f"파일 크기({total_rows}행)가 분할 기준({max_items}행) 이하로 분할 불필요")
            return [file_path]
            
        # 파일 분할 필요
        num_parts = (total_rows + max_items - 1) // max_items  # 올림 나눗셈
        logging.info(f"파일 분할 필요: {total_rows}행 -> {num_parts}개 파일")
        
        # 파일 이름 및 경로 구성
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_path)
        file_base, file_ext = os.path.splitext(file_name)
        
        output_files = []
        
        # 파일 분할 저장
        for i in range(num_parts):
            start_idx = i * max_items
            end_idx = min((i + 1) * max_items, total_rows)
            
            part_df = df.iloc[start_idx:end_idx]
            
            # 출력 파일명: 원본파일명_part1(300)
            part_file_name = f"{file_base}_part{i+1}({end_idx-start_idx}){file_ext}"
            part_file_path = os.path.join(file_dir, part_file_name)
            
            try:
                part_df.to_excel(part_file_path, index=False)
                output_files.append(part_file_path)
                logging.info(f"분할 파일 생성: {part_file_path} ({end_idx-start_idx}행)")
            except Exception as save_err:
                logging.error(f"분할 파일 저장 중 오류: {str(save_err)}")
                # 저장 실패해도 계속 진행
            
        # 하나 이상의 파일이 성공적으로 생성되었는지 확인
        if output_files:
            return output_files
        else:
            logging.error("모든 분할 파일 생성 실패")
            return [file_path]  # 실패 시 원본 경로 반환
        
    except Exception as e:
        logging.error(f"파일 분할 처리 중 오류 발생: {str(e)}")
        return [file_path]  # 오류 발생 시 원본 파일 반환

def main():
    """메인 함수"""
    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description="쇼핑 자동화 프로그램")
    parser.add_argument("--use_proxies", action="store_true", help="프록시 사용 여부")
    parser.add_argument('--cli', action='store_true', help='CLI 모드 실행 (GUI 없이)')
    parser.add_argument('--input-files', nargs='+', help='CLI 모드에서 처리할 Excel 파일 목록')
    parser.add_argument('--output-dir', help='결과 파일을 저장할 디렉토리')
    parser.add_argument('--limit', type=int, help='처리할 상품 수 제한 (기본값: 전체 상품)')
    parser.add_argument('--split', action='store_true', help='작업메뉴얼 기준 대용량 파일 자동 분할 (300행 단위)')
    parser.add_argument('--manual-workflow', action='store_true', help='작업메뉴얼 지침에 따른 전체 워크플로우 실행')
    parser.add_argument('--no-email', action='store_true', help='이메일 전송 비활성화 (작업메뉴얼 워크플로우에서)')
    parser.add_argument('--no-second-stage', action='store_true', help='2차 파일 생성 비활성화 (작업메뉴얼 워크플로우에서)')
    parser.add_argument('--email-recipient', default='dasomas@kakao.com', help='이메일 수신자 (기본값: dasomas@kakao.com)')
    parser.add_argument('--help-manual', action='store_true', help='작업메뉴얼 워크플로우 사용법 출력')
    args = parser.parse_args()
    
    # 작업메뉴얼 워크플로우 도움말 출력
    if args.help_manual:
        print(MANUAL_WORKFLOW_HELP)
        sys.exit(0)
    
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
            # 입력 파일 확인
            for input_file in args.input_files:
                if not os.path.exists(input_file):
                    logger.error(f"입력 파일이 존재하지 않습니다: {input_file}")
                    print(f"오류: 입력 파일이 존재하지 않습니다: {input_file}")
                    sys.exit(1)
        
            # 프로세서 초기화
            processor = ProductProcessor(config)
            
            all_input_files = []
            
            # 파일 분할 처리 (필요시)
            if args.split:
                for input_file in args.input_files:
                    try:
                        split_files = split_large_file(input_file)
                        all_input_files.extend(split_files)
                        
                        if len(split_files) > 1:
                            logger.info(f"파일 분할 완료: {input_file} -> {len(split_files)}개 파일")
                    except Exception as split_err:
                        logger.error(f"파일 분할 중 오류 발생: {str(split_err)}")
                        all_input_files.append(input_file)  # 원본 파일 사용
            else:
                all_input_files = args.input_files
            
            # 출력 디렉토리 확인 및 생성
            if args.output_dir and not os.path.exists(args.output_dir):
                try:
                    os.makedirs(args.output_dir, exist_ok=True)
                    logger.info(f"출력 디렉토리 생성: {args.output_dir}")
                except Exception as dir_err:
                    logger.error(f"출력 디렉토리 생성 실패: {str(dir_err)}")
                    print(f"오류: 출력 디렉토리 생성 실패: {str(dir_err)}")
                    sys.exit(1)
                
            # 작업메뉴얼 워크플로우 실행
            if args.manual_workflow:
                results = []
                for input_file in all_input_files:
                    logger.info(f"작업메뉴얼 워크플로우 실행: {input_file}")
                    try:
                        result = processor.run_workflow_from_manual(
                            input_file=input_file,
                            output_dir=args.output_dir,
                            send_email=not args.no_email,
                            generate_second_stage=not args.no_second_stage,
                            email_recipient=args.email_recipient
                        )
                        results.append(result)
                    except Exception as workflow_err:
                        logger.error(f"워크플로우 실행 중 오류: {str(workflow_err)}")
                        results.append({
                            "input_file": input_file,
                            "status": "error",
                            "error": str(workflow_err)
                        })
                    
                # 결과 요약
                success_count = sum(1 for r in results if r.get('status') == 'success')
                print(f"워크플로우 완료: {success_count}/{len(results)}개 파일 성공")
                for result in results:
                    status = "✅ 성공" if result.get('status') == 'success' else "❌ 실패"
                    input_file = os.path.basename(result.get('input_file', ''))
                    print(f"- {input_file}: {status}")
                    if result.get('status') == 'success':
                        if result.get('first_stage_file'):
                            print(f"  1차 파일: {os.path.basename(result['first_stage_file'])}")
                        if result.get('second_stage_file'):
                            print(f"  2차 파일: {os.path.basename(result['second_stage_file'])}")
                        if result.get('email_sent'):
                            print(f"  이메일 전송: {result['email_recipient']}")
                    else:
                        print(f"  오류: {result.get('error', '알 수 없는 오류')}")
                        
                sys.exit(0)
            
            # 기존 처리 로직
            output_files = processor.process_files(all_input_files, args.output_dir, args.limit)
            
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