import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
import traceback
from typing import Optional
import shutil

def setup_logging(log_dir: Optional[str] = None) -> None:
    """Setup logging configuration with improved error handling and features."""
    try:
        # Create log directory if specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Get current timestamp for log file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'File: %(pathname)s:%(lineno)d\n'
            'Function: %(funcName)s\n'
            '%(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # Error console handler (ERROR level)
        error_console_handler = logging.StreamHandler(sys.stderr)
        error_console_handler.setLevel(logging.ERROR)
        error_console_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_console_handler)
        
        if log_dir:
            # File handler for all logs (DEBUG level)
            all_log_file = os.path.join(log_dir, f'all_{timestamp}.log')
            file_handler = RotatingFileHandler(
                all_log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file handler (ERROR level)
            error_log_file = os.path.join(log_dir, f'error_{timestamp}.log')
            error_file_handler = TimedRotatingFileHandler(
                error_log_file,
                when='midnight',
                interval=1,
                backupCount=30,  # Keep 30 days of error logs
                encoding='utf-8'
            )
            error_file_handler.setLevel(logging.ERROR)
            error_file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_file_handler)
            
            # Create a copy of the latest log file instead of symlink
            latest_log = os.path.join(log_dir, 'latest.log')
            if os.path.exists(latest_log):
                os.remove(latest_log)
            shutil.copy2(all_log_file, latest_log)
        
        # Add unhandled exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupts
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
            else:
                root_logger.error(
                    "Uncaught exception",
                    exc_info=(exc_type, exc_value, exc_traceback)
                )
        
        sys.excepthook = handle_exception
        
        # Log successful setup
        root_logger.info("Logging system initialized successfully")
        
    except Exception as e:
        # If logging setup fails, try to log to console as last resort
        print(f"Failed to setup logging: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise

class LogCapture:
    """Context manager for capturing log output during a specific operation."""
    def __init__(self, logger_name: str, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.handler = None
        self.records = []
        
    def __enter__(self):
        class RecordListHandler(logging.Handler):
            def __init__(self, records):
                super().__init__()
                self.records = records
                
            def emit(self, record):
                self.records.append(record)
        
        self.handler = RecordListHandler(self.records)
        self.handler.setLevel(self.level)
        self.logger.addHandler(self.handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            self.logger.removeHandler(self.handler)
            
    def get_records(self) -> list:
        """Get captured log records."""
        return self.records
        
    def get_messages(self) -> list:
        """Get formatted log messages."""
        return [self.handler.format(record) for record in self.records] 