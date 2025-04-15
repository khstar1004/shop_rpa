"""Handles loading and processing of configuration from .ini and .env files."""
import configparser
import os
from typing import Any, Dict

from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """Load configuration from config.ini file and .env file"""
    load_dotenv()

    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        config.read(config_path, encoding="utf-8")
    except Exception as e:
        raise Exception(f"Failed to read config file: {e}") from e

    # Convert string values to appropriate types
    processed_config = {}

    try:
        # API section
        processed_config["api"] = {
            "naver_client_id": os.getenv("NAVER_CLIENT_ID", config["API"].get("naver_client_id")),
            "naver_client_secret": os.getenv("NAVER_CLIENT_SECRET", config["API"].get("naver_client_secret")),
        }

        # MATCHING section
        processed_config["matching"] = {
            "text_similarity_threshold": float(config["MATCHING"]["text_similarity_threshold"]),
            "image_similarity_threshold": float(config["MATCHING"]["image_similarity_threshold"]),
            "text_weight": float(config["MATCHING"]["text_weight"]),
            "image_weight": float(config["MATCHING"]["image_weight"]),
        }

        # PROCESSING section
        processed_config["processing"] = {
            "max_workers": int(config["PROCESSING"]["max_workers"]),
            "max_retries": int(config["PROCESSING"]["max_retries"]),
            "cache_duration": int(config["PROCESSING"]["cache_duration"]),
            "cache_max_size_mb": int(config["PROCESSING"]["cache_max_size_mb"]),
            "request_timeout": int(config["PROCESSING"]["request_timeout"]),
            "batch_size": int(config["PROCESSING"]["batch_size"]),
            "memory_limit_mb": int(config["PROCESSING"]["memory_limit_mb"]),
            "enable_compression": config["PROCESSING"]["enable_compression"].lower() == "true",
            "compression_level": int(config["PROCESSING"]["compression_level"]),
            "auto_split_files": config["PROCESSING"]["auto_split_files"].lower() == "true",
            "split_threshold": int(config["PROCESSING"]["split_threshold"]),
            "auto_merge_results": config["PROCESSING"]["auto_merge_results"].lower() == "true",
            "auto_clean_product_names": config["PROCESSING"]["auto_clean_product_names"].lower() == "true",
        }

        # SCRAPING section
        processed_config["scraping"] = {
            "max_concurrent_requests": int(config["SCRAPING"]["max_concurrent_requests"]),
            "extraction_timeout": int(config["SCRAPING"]["extraction_timeout"]),
            "enable_dom_extraction": config["SCRAPING"]["enable_dom_extraction"].lower() == "true",
            "enable_text_extraction": config["SCRAPING"]["enable_text_extraction"].lower() == "true",
            "enable_coord_extraction": config["SCRAPING"]["enable_coord_extraction"].lower() == "true",
            "use_fallback_mechanism": config["SCRAPING"]["use_fallback_mechanism"].lower() == "true",
            "auto_detect_content_type": config["SCRAPING"]["auto_detect_content_type"].lower() == "true",
            "use_sparse_structures": config["SCRAPING"]["use_sparse_structures"].lower() == "true",
            "selective_dom_observation": config["SCRAPING"]["selective_dom_observation"].lower() == "true",
            "async_tasks": config["SCRAPING"]["async_tasks"].lower() == "true",
            "session_persistence": config["SCRAPING"]["session_persistence"].lower() == "true",
            "politeness_delay": int(config["SCRAPING"]["politeness_delay"]),
            "user_experience_priority": config["SCRAPING"]["user_experience_priority"].lower() == "true",
            "connection_pool_size": int(config["SCRAPING"]["connection_pool_size"]),
            "ssl_verification": config["SCRAPING"]["ssl_verification"].lower() == "true",
            "follow_redirects": config["SCRAPING"]["follow_redirects"].lower() == "true",
            "max_redirects": int(config["SCRAPING"]["max_redirects"]),
            "retry_on_network_error": config["SCRAPING"]["retry_on_network_error"].lower() == "true",
            "retry_on_specific_status": [int(x.strip()) for x in config["SCRAPING"]["retry_on_specific_status"].split(",")],
            "exponential_backoff": config["SCRAPING"]["exponential_backoff"].lower() == "true",
        }

        # EXCEL section
        processed_config["excel"] = {
            "default_sheet_name": config["EXCEL"]["default_sheet_name"],
            "alternative_sheet_names": [x.strip() for x in config["EXCEL"]["alternative_sheet_names"].split(",")],
            "start_row": int(config["EXCEL"]["start_row"]),
            "alternative_start_rows": [int(x.strip()) for x in config["EXCEL"]["alternative_start_rows"].split(",")],
            "required_columns": [x.strip() for x in config["EXCEL"]["required_columns"].split(",")],
            "column_alternatives": {
                "상품명": [x.strip() for x in config["EXCEL"]["column_alternatives_상품명"].split(",")],
                "판매단가": [x.strip() for x in config["EXCEL"]["column_alternatives_판매단가"].split(",")],
                "상품code": [x.strip() for x in config["EXCEL"]["column_alternatives_상품code"].split(",")],
                "이미지": [x.strip() for x in config["EXCEL"]["column_alternatives_이미지"].split(",")],
                "링크": [x.strip() for x in config["EXCEL"]["column_alternatives_링크"].split(",")],
            },
            "optional_columns": [x.strip() for x in config["EXCEL"]["optional_columns"].split(",")],
            "max_rows": int(config["EXCEL"]["max_rows"]),
            "max_file_size_mb": int(config["EXCEL"]["max_file_size_mb"]),
            "validation_rules": config["EXCEL"]["validation_rules"].lower() == "true",
            "price_min": int(config["EXCEL"]["price_min"]),
            "price_max": int(config["EXCEL"]["price_max"]),
            "product_code_pattern": config["EXCEL"]["product_code_pattern"],
            "url_pattern": config["EXCEL"]["url_pattern"],
            "enable_data_quality_metrics": config["EXCEL"]["enable_data_quality_metrics"].lower() == "true",
            "enable_duplicate_detection": config["EXCEL"]["enable_duplicate_detection"].lower() == "true",
            "enable_auto_correction": config["EXCEL"]["enable_auto_correction"].lower() == "true",
            "auto_correction_rules": [x.strip() for x in config["EXCEL"]["auto_correction_rules"].split(",")],
            "report_formatting": config["EXCEL"]["report_formatting"].lower() == "true",
            "report_styles": config["EXCEL"]["report_styles"].lower() == "true",
            "report_filters": config["EXCEL"]["report_filters"].lower() == "true",
            "report_sorting": config["EXCEL"]["report_sorting"].lower() == "true",
            "report_freeze_panes": config["EXCEL"]["report_freeze_panes"].lower() == "true",
            "report_auto_fit": config["EXCEL"]["report_auto_fit"].lower() == "true",
            "attempt_all_sheets": config["EXCEL"]["attempt_all_sheets"].lower() == "true",
            "flexible_column_mapping": config["EXCEL"]["flexible_column_mapping"].lower() == "true",
            "create_missing_columns": config["EXCEL"]["create_missing_columns"].lower() == "true",
            "enable_formatting": config["EXCEL"]["enable_formatting"].lower() == "true",
            "date_format": config["EXCEL"]["date_format"],
            "number_format": config["EXCEL"]["number_format"],
        }

        # PATHS section
        processed_config["paths"] = {
            "cache_dir": os.path.abspath(config["PATHS"]["cache_dir"]),
            "output_dir": os.path.abspath(config["PATHS"]["output_dir"]),
            "log_dir": os.path.abspath(config["PATHS"]["log_dir"]),
            "temp_dir": os.path.abspath(config["PATHS"]["temp_dir"]),
            "backup_dir": os.path.abspath(config["PATHS"]["backup_dir"]),
            "intermediate_dir": os.path.abspath(config["PATHS"]["intermediate_dir"]),
            "final_dir": os.path.abspath(config["PATHS"]["final_dir"]),
        }

        # GUI section - keep as ConfigParser object
        processed_config["gui_config"] = config

    except Exception as e:
        raise Exception(f"Failed to process config values: {e}") from e

    return processed_config
