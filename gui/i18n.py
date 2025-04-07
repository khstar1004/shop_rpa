"""Internationalization (i18n) support module for the application"""

import json
import os
import logging
from pathlib import Path
from typing import Dict

class Translator:
    """다국어 지원을 위한 번역 관리 클래스"""
    
    def __init__(self):
        """Initialize the translator"""
        self.current_language = "ko_KR"
        self.translations = {}
        self.translations_dir = Path(__file__).parent / "translations"
        self.load_translations()
    
    def load_translations(self):
        """번역 파일들을 로드합니다."""
        try:
            if not self.translations_dir.exists():
                self.translations_dir.mkdir(parents=True)
                self._create_default_translations()
            
            for lang_file in self.translations_dir.glob("*.json"):
                lang_code = lang_file.stem
                try:
                    with open(lang_file, "r", encoding="utf-8") as f:
                        self.translations[lang_code] = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"번역 파일 '{lang_file}'이 손상되었습니다.")
                    continue
                except Exception as e:
                    logger.error(f"번역 파일 '{lang_file}' 로드 중 오류 발생: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"번역 파일 로드 중 오류 발생: {str(e)}")
            self._create_default_translations()
    
    def _create_default_translations(self):
        """기본 번역 파일을 생성합니다."""
        default_translations = {
            "ko_KR": {
                "app_title": "Shop RPA",
                "memory_usage": "메모리 사용량: {memory:.1f} MB",
                "status_ready": "준비",
                "status_processing": "처리 중...",
                "status_completed": "완료",
                "status_error": "오류",
                "error_occurred": "오류가 발생했습니다: {error}",
                "confirm_exit": "프로그램을 종료하시겠습니까?",
                "yes": "예",
                "no": "아니오",
                "cancel": "취소",
                "save": "저장",
                "load": "불러오기",
                "settings": "설정",
                "language": "언어",
                "theme": "테마",
                "dark_mode": "다크 모드",
                "light_mode": "라이트 모드",
                "auto_save": "자동 저장",
                "thread_count": "스레드 수",
                "max_log_lines": "최대 로그 라인 수",
                "log_level": "로그 레벨",
                "debug": "디버그",
                "info": "정보",
                "warning": "경고",
                "error": "오류",
                "critical": "치명적"
            },
            "en_US": {
                "app_title": "Shop RPA",
                "memory_usage": "Memory Usage: {memory:.1f} MB",
                "status_ready": "Ready",
                "status_processing": "Processing...",
                "status_completed": "Completed",
                "status_error": "Error",
                "error_occurred": "An error occurred: {error}",
                "confirm_exit": "Do you want to exit the program?",
                "yes": "Yes",
                "no": "No",
                "cancel": "Cancel",
                "save": "Save",
                "load": "Load",
                "settings": "Settings",
                "language": "Language",
                "theme": "Theme",
                "dark_mode": "Dark Mode",
                "light_mode": "Light Mode",
                "auto_save": "Auto Save",
                "thread_count": "Thread Count",
                "max_log_lines": "Max Log Lines",
                "log_level": "Log Level",
                "debug": "Debug",
                "info": "Info",
                "warning": "Warning",
                "error": "Error",
                "critical": "Critical"
            }
        }
        
        for lang_code, translations in default_translations.items():
            lang_file = self.translations_dir / f"{lang_code}.json"
            try:
                with open(lang_file, "w", encoding="utf-8") as f:
                    json.dump(translations, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.error(f"기본 번역 파일 '{lang_file}' 생성 중 오류 발생: {str(e)}")
    
    def set_language(self, lang_code: str) -> bool:
        """현재 언어를 설정합니다."""
        if lang_code in self.translations:
            self.current_language = lang_code
            return True
        return False
    
    def get_text(self, key: str, default: str = None, **kwargs) -> str:
        """번역된 텍스트를 반환합니다.
        
        Args:
            key: 번역 키
            default: 기본값 (키가 없을 경우 반환할 값)
            **kwargs: 텍스트 포맷팅에 사용할 매개변수
            
        Returns:
            번역된 텍스트 또는 기본값
        """
        try:
            text = self.translations.get(self.current_language, {}).get(key, default if default is not None else key)
            if kwargs:
                try:
                    return text.format(**kwargs)
                except KeyError as e:
                    logger.error(f"번역 텍스트 포맷팅 중 오류 발생: {str(e)}")
                    return text
            return text
        except Exception as e:
            logger.error(f"텍스트 '{key}' 가져오기 중 오류 발생: {str(e)}")
            return default if default is not None else key
    
    def get_available_languages(self) -> Dict[str, str]:
        """사용 가능한 언어 목록을 반환합니다."""
        return {
            "ko_KR": "한국어",
            "en_US": "English"
        }

# Global translator instance
translator = Translator() 