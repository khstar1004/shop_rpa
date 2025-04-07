"""Unit tests for the Settings module"""

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

from gui.settings import Settings

class TestSettings(unittest.TestCase):
    """Test cases for the Settings class"""
    
    def setUp(self):
        """Set up test fixtures, if any"""
        # Create a temporary directory for test settings
        self.test_dir = tempfile.mkdtemp()
        
        # Patch os.path.expanduser to use our test directory
        self.patcher = patch('gui.settings.os.path.expanduser')
        self.mock_expanduser = self.patcher.start()
        self.mock_expanduser.return_value = self.test_dir
        
        # Create a settings instance
        self.settings = Settings()
    
    def tearDown(self):
        """Tear down test fixtures, if any"""
        # Stop patchers
        self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_default_settings(self):
        """Test default settings are created correctly"""
        # Create a new settings instance
        settings = Settings()
        
        # Check default values
        self.assertFalse(settings.get("dark_mode"))
        self.assertTrue(settings.get("auto_save"))
        self.assertEqual(settings.get("thread_count"), 4)
        self.assertEqual(settings.get("auto_save_interval"), 300)
        self.assertEqual(settings.get("language"), "ko_KR")
    
    def test_save_and_load_settings(self):
        """Test saving and loading settings"""
        # Change some settings
        self.settings.set("dark_mode", True)
        self.settings.set("thread_count", 8)
        self.settings.set("language", "en_US")
        
        # Save settings
        self.settings.save_settings()
        
        # Create a new settings instance that should load from file
        new_settings = Settings()
        
        # Check the loaded values match what we saved
        self.assertTrue(new_settings.get("dark_mode"))
        self.assertEqual(new_settings.get("thread_count"), 8)
        self.assertEqual(new_settings.get("language"), "en_US")
    
    def test_get_nonexistent_setting(self):
        """Test getting a setting that doesn't exist"""
        # Default value should be returned for nonexistent setting
        value = self.settings.get("nonexistent_setting", "default_value")
        self.assertEqual(value, "default_value")
    
    def test_reset_settings(self):
        """Test resetting settings to defaults"""
        # Change some settings
        self.settings.set("dark_mode", True)
        self.settings.set("thread_count", 8)
        
        # Reset settings
        self.settings.reset_to_defaults()
        
        # Check values were reset
        self.assertFalse(self.settings.get("dark_mode"))
        self.assertEqual(self.settings.get("thread_count"), 4)
    
    @patch('gui.settings.logging.error')
    def test_corrupt_settings_file(self, mock_logging):
        """Test handling of corrupt settings file"""
        # Create a corrupt settings file
        settings_file = os.path.join(self.test_dir, 'shoprpa_settings.json')
        with open(settings_file, 'w') as f:
            f.write("This is not valid JSON")
        
        # Create a new settings instance that should handle the corrupt file
        new_settings = Settings()
        
        # Check that error was logged
        mock_logging.assert_called()
        
        # Check that default settings were used
        self.assertFalse(new_settings.get("dark_mode"))
        self.assertEqual(new_settings.get("thread_count"), 4)
    
    def test_get_all_settings(self):
        """Test getting all settings"""
        # Change some settings
        self.settings.set("dark_mode", True)
        self.settings.set("custom_setting", "custom_value")
        
        # Get all settings
        all_settings = self.settings.get_all_settings()
        
        # Check the settings dict contains our changes
        self.assertTrue(all_settings["dark_mode"])
        self.assertEqual(all_settings["custom_setting"], "custom_value")
    
    def test_log_level(self):
        """Test log level methods"""
        # Default log level should be INFO
        self.assertEqual(self.settings.get_log_level(), logging.INFO)
        
        # Change log level to DEBUG
        self.settings.set("log_level", "DEBUG")
        self.assertEqual(self.settings.get_log_level(), logging.DEBUG)
        
        # Test invalid log level
        self.settings.set("log_level", "INVALID")
        self.assertEqual(self.settings.get_log_level(), logging.INFO)

if __name__ == '__main__':
    unittest.main() 