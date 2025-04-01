import configparser
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.ini file"""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        config.read(config_path, encoding='utf-8')
    except Exception as e:
        raise Exception(f"Failed to read config file: {e}")
    
    # Convert string values to appropriate types
    processed_config = {}
    
    try:
        # API section
        processed_config['API'] = dict(config['API'])
        
        # MATCHING section
        processed_config['MATCHING'] = {
            'TEXT_SIMILARITY_THRESHOLD': float(config['MATCHING']['TEXT_SIMILARITY_THRESHOLD']),
            'IMAGE_SIMILARITY_THRESHOLD': float(config['MATCHING']['IMAGE_SIMILARITY_THRESHOLD']),
            'TEXT_WEIGHT': float(config['MATCHING']['TEXT_WEIGHT']),
            'IMAGE_WEIGHT': float(config['MATCHING']['IMAGE_WEIGHT'])
        }
        
        # PROCESSING section
        processed_config['PROCESSING'] = {
            'MAX_WORKERS': int(config['PROCESSING']['MAX_WORKERS']),
            'MAX_RETRIES': int(config['PROCESSING']['MAX_RETRIES']),
            'CACHE_DURATION': int(config['PROCESSING']['CACHE_DURATION']),
            'CACHE_MAX_SIZE_MB': int(config['PROCESSING']['CACHE_MAX_SIZE_MB']),
            'REQUEST_TIMEOUT': int(config['PROCESSING']['REQUEST_TIMEOUT']),
            'BATCH_SIZE': int(config['PROCESSING']['BATCH_SIZE']),
            'MEMORY_LIMIT_MB': int(config['PROCESSING']['MEMORY_LIMIT_MB']),
            'ENABLE_COMPRESSION': config['PROCESSING']['ENABLE_COMPRESSION'].lower() == 'true',
            'COMPRESSION_LEVEL': int(config['PROCESSING']['COMPRESSION_LEVEL'])
        }
        
        # Add new settings for file handling based on manual workflow
        if 'FILE_HANDLING' in config:
            processed_config['PROCESSING']['AUTO_SPLIT_FILES'] = config['FILE_HANDLING'].get('AUTO_SPLIT_FILES', 'true').lower() == 'true'
            processed_config['PROCESSING']['SPLIT_THRESHOLD'] = int(config['FILE_HANDLING'].get('SPLIT_THRESHOLD', '300'))
            processed_config['PROCESSING']['AUTO_MERGE_RESULTS'] = config['FILE_HANDLING'].get('AUTO_MERGE_RESULTS', 'true').lower() == 'true'
            processed_config['PROCESSING']['AUTO_CLEAN_PRODUCT_NAMES'] = config['FILE_HANDLING'].get('AUTO_CLEAN_PRODUCT_NAMES', 'true').lower() == 'true'
        else:
            # Default values if section doesn't exist
            processed_config['PROCESSING']['AUTO_SPLIT_FILES'] = True
            processed_config['PROCESSING']['SPLIT_THRESHOLD'] = 300
            processed_config['PROCESSING']['AUTO_MERGE_RESULTS'] = True
            processed_config['PROCESSING']['AUTO_CLEAN_PRODUCT_NAMES'] = True

        # Add price comparison settings from manual
        if 'PRICE_COMPARISON' in config:
            processed_config['PROCESSING']['MIN_PRICE_DIFF_PERCENT'] = float(config['PRICE_COMPARISON'].get('MIN_PRICE_DIFF_PERCENT', '10.0'))
            processed_config['PROCESSING']['HIGHLIGHT_PRICE_DIFF'] = config['PRICE_COMPARISON'].get('HIGHLIGHT_PRICE_DIFF', 'true').lower() == 'true'
        else:
            # Default values if section doesn't exist
            processed_config['PROCESSING']['MIN_PRICE_DIFF_PERCENT'] = 10.0
            processed_config['PROCESSING']['HIGHLIGHT_PRICE_DIFF'] = True
        
        # SCRAPING section
        if 'SCRAPING' in config:
            processed_config['SCRAPING'] = {
                'MAX_CONCURRENT_REQUESTS': int(config['SCRAPING']['MAX_CONCURRENT_REQUESTS']),
                'EXTRACTION_TIMEOUT': int(config['SCRAPING']['EXTRACTION_TIMEOUT']),
                'ENABLE_DOM_EXTRACTION': config['SCRAPING']['ENABLE_DOM_EXTRACTION'].lower() == 'true',
                'ENABLE_TEXT_EXTRACTION': config['SCRAPING']['ENABLE_TEXT_EXTRACTION'].lower() == 'true',
                'ENABLE_COORD_EXTRACTION': config['SCRAPING']['ENABLE_COORD_EXTRACTION'].lower() == 'true',
                'USE_FALLBACK_MECHANISM': config['SCRAPING']['USE_FALLBACK_MECHANISM'].lower() == 'true',
                'AUTO_DETECT_CONTENT_TYPE': config['SCRAPING']['AUTO_DETECT_CONTENT_TYPE'].lower() == 'true',
                'USE_SPARSE_STRUCTURES': config['SCRAPING']['USE_SPARSE_STRUCTURES'].lower() == 'true',
                'SELECTIVE_DOM_OBSERVATION': config['SCRAPING']['SELECTIVE_DOM_OBSERVATION'].lower() == 'true',
                'ASYNC_TASKS': config['SCRAPING']['ASYNC_TASKS'].lower() == 'true',
                'SESSION_PERSISTENCE': config['SCRAPING']['SESSION_PERSISTENCE'].lower() == 'true',
                'POLITENESS_DELAY': int(config['SCRAPING']['POLITENESS_DELAY']),
                'USER_EXPERIENCE_PRIORITY': config['SCRAPING']['USER_EXPERIENCE_PRIORITY'].lower() == 'true',
                'CONNECTION_POOL_SIZE': int(config['SCRAPING']['CONNECTION_POOL_SIZE']),
                'SSL_VERIFICATION': config['SCRAPING']['SSL_VERIFICATION'].lower() == 'true',
                'FOLLOW_REDIRECTS': config['SCRAPING']['FOLLOW_REDIRECTS'].lower() == 'true',
                'MAX_REDIRECTS': int(config['SCRAPING']['MAX_REDIRECTS']),
                'RETRY_ON_NETWORK_ERROR': config['SCRAPING']['RETRY_ON_NETWORK_ERROR'].lower() == 'true',
                'RETRY_ON_SPECIFIC_STATUS': [int(x.strip()) for x in config['SCRAPING']['RETRY_ON_SPECIFIC_STATUS'].split(',')],
                'EXPONENTIAL_BACKOFF': config['SCRAPING']['EXPONENTIAL_BACKOFF'].lower() == 'true'
            }
            
            # Add Naver search settings as per manual
            processed_config['SCRAPING']['MAX_PAGES'] = int(config['SCRAPING'].get('MAX_PAGES', '3'))
            processed_config['SCRAPING']['REQUIRE_IMAGE_MATCH'] = config['SCRAPING'].get('REQUIRE_IMAGE_MATCH', 'true').lower() == 'true'
        
        # EXCEL section
        if 'EXCEL' in config:
            processed_config['EXCEL'] = {
                'SHEET_NAME': config['EXCEL']['SHEET_NAME'],
                'START_ROW': int(config['EXCEL']['START_ROW']),
                'REQUIRED_COLUMNS': [x.strip() for x in config['EXCEL']['REQUIRED_COLUMNS'].split(',')],
                'OPTIONAL_COLUMNS': [x.strip() for x in config['EXCEL']['OPTIONAL_COLUMNS'].split(',')],
                'MAX_ROWS': int(config['EXCEL']['MAX_ROWS']),
                'ENABLE_FORMATTING': config['EXCEL']['ENABLE_FORMATTING'].lower() == 'true',
                'DATE_FORMAT': config['EXCEL']['DATE_FORMAT'],
                'NUMBER_FORMAT': config['EXCEL']['NUMBER_FORMAT'],
                'MAX_FILE_SIZE_MB': int(config['EXCEL']['MAX_FILE_SIZE_MB']),
                'VALIDATION_RULES': config['EXCEL']['VALIDATION_RULES'].lower() == 'true',
                'PRICE_MIN': int(config['EXCEL']['PRICE_MIN']),
                'PRICE_MAX': int(config['EXCEL']['PRICE_MAX']),
                'PRODUCT_CODE_PATTERN': config['EXCEL']['PRODUCT_CODE_PATTERN'],
                'URL_PATTERN': config['EXCEL']['URL_PATTERN'],
                'ENABLE_DATA_QUALITY_METRICS': config['EXCEL']['ENABLE_DATA_QUALITY_METRICS'].lower() == 'true',
                'ENABLE_DUPLICATE_DETECTION': config['EXCEL']['ENABLE_DUPLICATE_DETECTION'].lower() == 'true',
                'ENABLE_AUTO_CORRECTION': config['EXCEL']['ENABLE_AUTO_CORRECTION'].lower() == 'true',
                'AUTO_CORRECTION_RULES': [x.strip() for x in config['EXCEL']['AUTO_CORRECTION_RULES'].split(',')],
                'REPORT_FORMATTING': config['EXCEL']['REPORT_FORMATTING'].lower() == 'true',
                'REPORT_STYLES': config['EXCEL']['REPORT_STYLES'].lower() == 'true',
                'REPORT_FILTERS': config['EXCEL']['REPORT_FILTERS'].lower() == 'true',
                'REPORT_SORTING': config['EXCEL']['REPORT_SORTING'].lower() == 'true',
                'REPORT_FREEZE_PANES': config['EXCEL']['REPORT_FREEZE_PANES'].lower() == 'true',
                'REPORT_AUTO_FIT': config['EXCEL']['REPORT_AUTO_FIT'].lower() == 'true'
            }
        
        # PATHS section
        processed_config['PATHS'] = {
            'CACHE_DIR': os.path.abspath(config['PATHS']['CACHE_DIR']),
            'OUTPUT_DIR': os.path.abspath(config['PATHS']['OUTPUT_DIR']),
            'LOG_DIR': os.path.abspath(config['PATHS']['LOG_DIR'])
        }
        
        # GUI section
        processed_config['GUI'] = {
            'WINDOW_WIDTH': int(config['GUI']['WINDOW_WIDTH']),
            'WINDOW_HEIGHT': int(config['GUI']['WINDOW_HEIGHT']),
            'MAX_LOG_LINES': int(config['GUI']['MAX_LOG_LINES']),
            'ENABLE_DARK_MODE': config['GUI']['ENABLE_DARK_MODE'].lower() == 'true',
            'SHOW_PROGRESS_BAR': config['GUI']['SHOW_PROGRESS_BAR'].lower() == 'true',
            'AUTO_SAVE_INTERVAL': int(config['GUI']['AUTO_SAVE_INTERVAL'])
        }
        
    except Exception as e:
        raise Exception(f"Error processing config values: {e}")
    
    return processed_config 