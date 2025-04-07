"""Unit tests for the i18n module"""

import unittest
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
import sys
import logging

# Add project root to path for imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.i18n import Translator

class TestTranslator(unittest.TestCase):
    """Test cases for the Translator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test translations
        self.test_dir = tempfile.mkdtemp()
        self.translations_dir = os.path.join(self.test_dir, "translations")
        os.makedirs(self.translations_dir, exist_ok=True)
        
        # Create test translation files
        self.create_test_translations()
        
        # Patch Path to use our test directory for translations
        self.patcher = patch('gui.i18n.Path')
        self.mock_path = self.patcher.start()
        
        # Setup the path mock to return our test directory for the translations
        mock_path_instance = MagicMock()
        mock_path_instance.parent = self.test_dir
        mock_path_instance.__truediv__.return_value = self.translations_dir
        self.mock_path.return_value = mock_path_instance
        
        # Make translations_dir.exists() return True
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        
        # Setup glob to return our test translation files
        ko_file = MagicMock()
        ko_file.stem = "ko_KR"
        ko_file.open = lambda mode, encoding: open(os.path.join(self.translations_dir, "ko_KR.json"), mode, encoding=encoding)
        
        en_file = MagicMock()
        en_file.stem = "en_US"
        en_file.open = lambda mode, encoding: open(os.path.join(self.translations_dir, "en_US.json"), mode, encoding=encoding)
        
        mock_dir.glob.return_value = [ko_file, en_file]
        self.mock_path.return_value.__truediv__.return_value = mock_dir
        
        # Create a translator instance
        self.translator = Translator()
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Stop patchers
        self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_test_translations(self):
        """Create test translation files"""
        # Korean translations
        ko_translations = {
            "file_selection": "파일 선택",
            "start": "시작",
            "stop": "중지",
            "language": "언어"
        }
        
        with open(os.path.join(self.translations_dir, "ko_KR.json"), "w", encoding="utf-8") as f:
            json.dump(ko_translations, f, ensure_ascii=False)
        
        # English translations
        en_translations = {
            "file_selection": "File Selection",
            "start": "Start",
            "stop": "Stop",
            "language": "Language"
        }
        
        with open(os.path.join(self.translations_dir, "en_US.json"), "w", encoding="utf-8") as f:
            json.dump(en_translations, f, ensure_ascii=False)
    
    def test_default_language(self):
        """Test default language is set correctly"""
        self.assertEqual(self.translator.current_language, "ko_KR")
    
    def test_get_text(self):
        """Test retrieving translated text"""
        # Test Korean text (default language)
        self.assertEqual(self.translator.get_text("file_selection"), "파일 선택")
        self.assertEqual(self.translator.get_text("start"), "시작")
        
        # Change language to English
        self.translator.set_language("en_US")
        
        # Test English text
        self.assertEqual(self.translator.get_text("file_selection"), "File Selection")
        self.assertEqual(self.translator.get_text("start"), "Start")
    
    def test_format_text(self):
        """Test formatting translated text with parameters"""
        # Add a test string with format placeholders to translations
        self.translator.translations["ko_KR"]["welcome_user"] = "안녕하세요, {name}님!"
        self.translator.translations["en_US"]["welcome_user"] = "Welcome, {name}!"
        
        # Test formatting with Korean
        self.assertEqual(
            self.translator.get_text("welcome_user", name="홍길동"),
            "안녕하세요, 홍길동님!"
        )
        
        # Test formatting with English
        self.translator.set_language("en_US")
        self.assertEqual(
            self.translator.get_text("welcome_user", name="John"),
            "Welcome, John!"
        )
    
    def test_nonexistent_key(self):
        """Test behavior with nonexistent translation key"""
        # Key doesn't exist, should return the key itself
        self.assertEqual(self.translator.get_text("nonexistent_key"), "nonexistent_key")
        
        # Key doesn't exist but default provided
        self.assertEqual(
            self.translator.get_text("nonexistent_key", default="Default Text"),
            "Default Text"
        )
    
    def test_set_nonexistent_language(self):
        """Test setting a language that doesn't exist"""
        # Save original language
        original_language = self.translator.current_language
        
        # Try to set a nonexistent language
        self.translator.set_language("fr_FR")
        
        # Language should not change
        self.assertEqual(self.translator.current_language, original_language)
    
    def test_available_languages(self):
        """Test getting available languages"""
        languages = self.translator.get_available_languages()
        self.assertIn("ko_KR", languages)
        self.assertIn("en_US", languages)
        self.assertEqual(languages["ko_KR"], "한국어")
        self.assertEqual(languages["en_US"], "English (US)")

if __name__ == '__main__':
    unittest.main() 