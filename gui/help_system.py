"""Help system"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QHBoxLayout
from .help_tip import HelpTipWidget
from .tutorial_dialog import TutorialDialog, TutorialStep
from .searchable_help import SearchableHelpText
from .help_constants import HELP_CONTENT, CONTEXT_HELP

class HelpSystem:
    """Help system for managing help content and interactions"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.help_tip = HelpTipWidget(main_window)
        self.context_help_enabled = False
        self._load_help_content()
    
    def _load_help_content(self):
        """Load help content"""
        self.help_content = HELP_CONTENT
        self.context_help = CONTEXT_HELP
    
    def show_context_help(self, widget, key):
        """Show context-sensitive help for a widget"""
        if not self.context_help_enabled:
            return
        
        if key in self.context_help:
            self.help_tip.show_tip(widget, self.context_help[key])
    
    def _get_context_help(self, key):
        """Get context help text"""
        return self.context_help.get(key, "")
    
    def toggle_context_help(self, enabled):
        """Toggle context help display"""
        self.context_help_enabled = enabled
    
    def show_tutorial(self):
        """Show interactive tutorial"""
        steps = [
            TutorialStep(
                "시작하기",
                "Shop RPA 시스템에 오신 것을 환영합니다. 이 프로그램은 상품 가격 비교를 자동화합니다.",
                self.main_window.file_group,
                "gui/assets/help/getting_started.png"
            ),
            TutorialStep(
                "파일 선택",
                "파일을 선택하는 방법에는 여러 가지가 있습니다:\n"
                "1. 파일 열기 버튼 클릭\n"
                "2. 파일을 드래그하여 드롭 영역에 놓기\n"
                "3. 최근 파일 목록에서 선택\n"
                "4. 즐겨찾기에서 선택",
                self.main_window.drop_area,
                "gui/assets/help/file_selection.png"
            ),
            TutorialStep(
                "설정",
                "설정 탭에서는 다양한 옵션을 조정할 수 있습니다:\n"
                "- 일반 설정: 언어, 로그 수준 등\n"
                "- 모양 및 테마: 다크 모드, UI 설정\n"
                "- 처리 설정: 스레드 수, 유사도 임계값 등",
                self.main_window.settings_tab,
                "gui/assets/help/settings.png"
            ),
            TutorialStep(
                "처리",
                "파일 처리는 다음과 같은 단계로 진행됩니다:\n"
                "1. 파일 로드 및 검증\n"
                "2. 데이터 전처리\n"
                "3. 가격 비교 분석\n"
                "4. 결과 저장",
                self.main_window.process_button,
                "gui/assets/help/processing.png"
            ),
            TutorialStep(
                "결과",
                "처리 결과는 다음과 같은 정보를 포함합니다:\n"
                "- 가격 비교 요약\n"
                "- 상세 분석 리포트\n"
                "- 추천 조치사항",
                self.main_window.results_area,
                "gui/assets/help/results.png"
            )
        ]
        
        tutorial = TutorialDialog(steps, self.main_window)
        tutorial.exec_()
    
    def show_help_dialog(self):
        """Show help dialog"""
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Help")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Help content
        help_widget = SearchableHelpText()
        help_widget.load_help_content(self.help_content)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        tutorial_button = QPushButton("Show Tutorial")
        tutorial_button.clicked.connect(self.show_tutorial)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        
        button_layout.addWidget(tutorial_button)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        
        layout.addWidget(help_widget)
        layout.addLayout(button_layout)
        
        dialog.exec_() 