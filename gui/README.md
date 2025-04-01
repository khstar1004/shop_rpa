# GUI 모듈

이 디렉토리는 쇼핑 RPA 시스템의 그래픽 사용자 인터페이스를 담당하는 모듈을 포함합니다.

## 구성 요소

- `main_window.py`: 메인 GUI 윈도우 구현

## 메인 윈도우 (main_window.py)

GUI 애플리케이션의 메인 윈도우를 정의합니다. 다음 기능을 제공합니다:

- 파일 드래그 앤 드롭 인터페이스
- 파일 선택 대화상자
- 처리 진행 상태 표시
- 로그 표시 영역
- 캐시 관리
- 백그라운드 처리 스레드
- 보고서 결과 표시:
  - 1차 및 2차 보고서 경로 표시
  - 결과 폴더 자동 열기
  - 오류 처리 및 사용자 알림

## 사용자 인터페이스 구성

- **파일 선택 섹션**: 입력 엑셀 파일 선택 기능
- **진행 상황 표시**: 작업 진행 상태 및 완료율 표시
- **설정 섹션**: 유사도 임계값, API 설정 등 구성
- **로그 출력 영역**: 처리 로그 실시간 표시
- **결과 영역**: 생성된 보고서 링크 및 열기 버튼

## 이벤트 처리

- `on_file_select()`: 파일 선택 이벤트 처리
- `on_process_start()`: 처리 시작 버튼 이벤트
- `on_process_complete()`: 처리 완료 이벤트
- `update_progress()`: 진행 상황 업데이트
- `open_report()`: 생성된 보고서 파일 열기

## 백그라운드 처리

GUI 응답성 유지를 위한 비동기 처리:

- `QThread` 사용하여 파일 처리 작업 백그라운드 실행
- 진행 상황 업데이트를 위한 시그널/슬롯 메커니즘
- 작업 완료 시 결과 업데이트 및 알림

## 사용 예시

```python
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 