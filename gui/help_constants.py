"""Help content constants"""

# Main help content
HELP_CONTENT = {
    "시작하기": {
        "text": "Shop RPA 시스템에 오신 것을 환영합니다. 이 프로그램은 상품 가격 비교를 자동화합니다.",
        "links": {"파일 선택": "file_selection", "설정": "settings"},
        "images": ["gui/assets/help/getting_started.png"]
    },
    "파일 선택": {
        "text": ("파일을 선택하는 방법에는 여러 가지가 있습니다:\n"
                 "1. 파일 열기 버튼 클릭\n"
                 "2. 파일을 드래그하여 드롭 영역에 놓기\n"
                 "3. 최근 파일 목록에서 선택\n"
                 "4. 즐겨찾기에서 선택"),
        "links": {},
        "images": ["gui/assets/help/file_selection.png"]
    },
    "설정": {
        "text": ("설정 탭에서는 다양한 옵션을 조정할 수 있습니다:\n"
                 "- 일반 설정: 언어, 로그 수준 등\n"
                 "- 모양 및 테마: 다크 모드, UI 설정\n"
                 "- 처리 설정: 스레드 수, 유사도 임계값 등"),
        "links": {},
        "images": ["gui/assets/help/settings.png"]
    },
    "처리": {
        "text": ("파일 처리는 다음과 같은 단계로 진행됩니다:\n"
                 "1. 파일 로드 및 검증\n"
                 "2. 데이터 전처리\n"
                 "3. 가격 비교 분석\n"
                 "4. 결과 저장"),
        "links": {},
        "images": ["gui/assets/help/processing.png"]
    },
    "결과": {
        "text": ("처리 결과는 다음과 같은 정보를 포함합니다:\n"
                 "- 가격 비교 요약\n"
                 "- 상세 분석 리포트\n"
                 "- 추천 조치사항"),
        "links": {},
        "images": ["gui/assets/help/results.png"]
    }
}

# Context-sensitive help
CONTEXT_HELP = {
    "file_open": "파일을 선택하거나 드래그하여 업로드하세요.",
    "file_drop": "파일을 여기에 드래그하여 업로드하세요.",
    "settings_general": "일반 설정을 조정하세요.",
    "settings_appearance": "UI 모양과 테마를 설정하세요.",
    "settings_processing": "데이터 처리 설정을 조정하세요.",
    "process_start": "파일 처리를 시작하세요.",
    "process_stop": "진행 중인 처리를 중지하세요.",
    "results_view": "처리 결과를 확인하세요.",
    "results_export": "결과를 파일로 내보내세요."
} 