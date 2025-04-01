import configparser
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.ini file"""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    config.read(config_path)
    
    # Convert string values to appropriate types
    processed_config = {}
    
    # API section
    processed_config['API'] = dict(config['API'])
    
    # MATCHING section
    processed_config['MATCHING'] = {
        'TEXT_SIMILARITY_THRESHOLD': config.getfloat('MATCHING', 'TEXT_SIMILARITY_THRESHOLD'),
        'IMAGE_SIMILARITY_THRESHOLD': config.getfloat('MATCHING', 'IMAGE_SIMILARITY_THRESHOLD'),
        'TEXT_WEIGHT': config.getfloat('MATCHING', 'TEXT_WEIGHT'),
        'IMAGE_WEIGHT': config.getfloat('MATCHING', 'IMAGE_WEIGHT')
    }
    
    # PROCESSING section
    processed_config['PROCESSING'] = {
        'MAX_WORKERS': config.getint('PROCESSING', 'MAX_WORKERS'),
        'MAX_RETRIES': config.getint('PROCESSING', 'MAX_RETRIES'),
        'CACHE_DURATION': config.getint('PROCESSING', 'CACHE_DURATION'),
        'CACHE_MAX_SIZE_MB': config.getint('PROCESSING', 'CACHE_MAX_SIZE_MB'),
        'REQUEST_TIMEOUT': config.getint('PROCESSING', 'REQUEST_TIMEOUT'),
        'BATCH_SIZE': config.getint('PROCESSING', 'BATCH_SIZE'),
        'MEMORY_LIMIT_MB': config.getint('PROCESSING', 'MEMORY_LIMIT_MB'),
        'ENABLE_COMPRESSION': config.getboolean('PROCESSING', 'ENABLE_COMPRESSION'),
        'COMPRESSION_LEVEL': config.getint('PROCESSING', 'COMPRESSION_LEVEL')
    }
    
    # SCRAPING section
    if 'SCRAPING' in config:
        processed_config['SCRAPING'] = {
            'MAX_CONCURRENT_REQUESTS': config.getint('SCRAPING', 'MAX_CONCURRENT_REQUESTS'),
            'EXTRACTION_TIMEOUT': config.getint('SCRAPING', 'EXTRACTION_TIMEOUT'),
            'ENABLE_DOM_EXTRACTION': config.getboolean('SCRAPING', 'ENABLE_DOM_EXTRACTION'),
            'ENABLE_TEXT_EXTRACTION': config.getboolean('SCRAPING', 'ENABLE_TEXT_EXTRACTION'),
            'ENABLE_COORD_EXTRACTION': config.getboolean('SCRAPING', 'ENABLE_COORD_EXTRACTION'),
            'USE_FALLBACK_MECHANISM': config.getboolean('SCRAPING', 'USE_FALLBACK_MECHANISM'),
            'AUTO_DETECT_CONTENT_TYPE': config.getboolean('SCRAPING', 'AUTO_DETECT_CONTENT_TYPE'),
            'USE_SPARSE_STRUCTURES': config.getboolean('SCRAPING', 'USE_SPARSE_STRUCTURES'),
            'SELECTIVE_DOM_OBSERVATION': config.getboolean('SCRAPING', 'SELECTIVE_DOM_OBSERVATION'),
            'ASYNC_TASKS': config.getboolean('SCRAPING', 'ASYNC_TASKS'),
            'SESSION_PERSISTENCE': config.getboolean('SCRAPING', 'SESSION_PERSISTENCE'),
            'POLITENESS_DELAY': config.getint('SCRAPING', 'POLITENESS_DELAY'),
            'USER_EXPERIENCE_PRIORITY': config.getboolean('SCRAPING', 'USER_EXPERIENCE_PRIORITY'),
            'CONNECTION_POOL_SIZE': config.getint('SCRAPING', 'CONNECTION_POOL_SIZE'),
            'SSL_VERIFICATION': config.getboolean('SCRAPING', 'SSL_VERIFICATION'),
            'FOLLOW_REDIRECTS': config.getboolean('SCRAPING', 'FOLLOW_REDIRECTS'),
            'MAX_REDIRECTS': config.getint('SCRAPING', 'MAX_REDIRECTS'),
            'RETRY_ON_NETWORK_ERROR': config.getboolean('SCRAPING', 'RETRY_ON_NETWORK_ERROR'),
            'RETRY_ON_SPECIFIC_STATUS': config.get('SCRAPING', 'RETRY_ON_SPECIFIC_STATUS').split(','),
            'EXPONENTIAL_BACKOFF': config.getboolean('SCRAPING', 'EXPONENTIAL_BACKOFF')
        }
    
    # EXCEL section
    if 'EXCEL' in config:
        processed_config['EXCEL'] = {
            'SHEET_NAME': config.get('EXCEL', 'SHEET_NAME'),
            'START_ROW': config.getint('EXCEL', 'START_ROW'),
            'REQUIRED_COLUMNS': config.get('EXCEL', 'REQUIRED_COLUMNS').split(','),
            'OPTIONAL_COLUMNS': config.get('EXCEL', 'OPTIONAL_COLUMNS').split(','),
            'MAX_ROWS': config.getint('EXCEL', 'MAX_ROWS'),
            'ENABLE_FORMATTING': config.getboolean('EXCEL', 'ENABLE_FORMATTING'),
            'DATE_FORMAT': config.get('EXCEL', 'DATE_FORMAT'),
            'NUMBER_FORMAT': config.get('EXCEL', 'NUMBER_FORMAT'),
            'MAX_FILE_SIZE_MB': config.getint('EXCEL', 'MAX_FILE_SIZE_MB'),
            'VALIDATION_RULES': config.getboolean('EXCEL', 'VALIDATION_RULES'),
            'PRICE_MIN': config.getint('EXCEL', 'PRICE_MIN'),
            'PRICE_MAX': config.getint('EXCEL', 'PRICE_MAX'),
            'PRODUCT_CODE_PATTERN': config.get('EXCEL', 'PRODUCT_CODE_PATTERN'),
            'URL_PATTERN': config.get('EXCEL', 'URL_PATTERN'),
            'ENABLE_DATA_QUALITY_METRICS': config.getboolean('EXCEL', 'ENABLE_DATA_QUALITY_METRICS'),
            'ENABLE_DUPLICATE_DETECTION': config.getboolean('EXCEL', 'ENABLE_DUPLICATE_DETECTION'),
            'ENABLE_AUTO_CORRECTION': config.getboolean('EXCEL', 'ENABLE_AUTO_CORRECTION'),
            'AUTO_CORRECTION_RULES': config.get('EXCEL', 'AUTO_CORRECTION_RULES').split(','),
            'REPORT_FORMATTING': config.getboolean('EXCEL', 'REPORT_FORMATTING'),
            'REPORT_STYLES': config.getboolean('EXCEL', 'REPORT_STYLES'),
            'REPORT_FILTERS': config.getboolean('EXCEL', 'REPORT_FILTERS'),
            'REPORT_SORTING': config.getboolean('EXCEL', 'REPORT_SORTING'),
            'REPORT_FREEZE_PANES': config.getboolean('EXCEL', 'REPORT_FREEZE_PANES'),
            'REPORT_AUTO_FIT': config.getboolean('EXCEL', 'REPORT_AUTO_FIT')
        }
    
    # PATHS section
    processed_config['PATHS'] = {
        'CACHE_DIR': os.path.abspath(config['PATHS']['CACHE_DIR']),
        'OUTPUT_DIR': os.path.abspath(config['PATHS']['OUTPUT_DIR']),
        'LOG_DIR': os.path.abspath(config['PATHS']['LOG_DIR'])
    }
    
    # GUI section
    processed_config['GUI'] = {
        'WINDOW_WIDTH': config.getint('GUI', 'WINDOW_WIDTH'),
        'WINDOW_HEIGHT': config.getint('GUI', 'WINDOW_HEIGHT'),
        'MAX_LOG_LINES': config.getint('GUI', 'MAX_LOG_LINES'),
        'ENABLE_DARK_MODE': config.getboolean('GUI', 'ENABLE_DARK_MODE', fallback=False),
        'SHOW_PROGRESS_BAR': config.getboolean('GUI', 'SHOW_PROGRESS_BAR', fallback=True),
        'AUTO_SAVE_INTERVAL': config.getint('GUI', 'AUTO_SAVE_INTERVAL', fallback=300)
    }
    
    return processed_config 