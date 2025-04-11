# Import refactored module structure while maintaining backward compatibility
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from utils.caching import FileCache, cache_result
from utils.reporting import generate_primary_report, generate_secondary_report

from .data_models import MatchResult, ProcessingResult, Product
from .matching.image_matcher import ImageMatcher
from .matching.multimodal_matcher import MultiModalMatcher
from .matching.text_matcher import TextMatcher

# Import the refactored Processor class
from .processing.main_processor import ProductProcessor as Processor
from .scraping.koryo_scraper import KoryoScraper
from .scraping.naver_crawler import NaverShoppingCrawler

# For backward compatibility, re-export these classes
__all__ = ["Processor", "Product", "MatchResult", "ProcessingResult"]

# Show deprecation warning
warnings.warn(
    "Direct import from 'processing' is deprecated. Use the new module structure instead: "
    "from core.processing import Processor",
    DeprecationWarning,
    stacklevel=2,
)

# Setup module logger
logger = logging.getLogger(__name__)
logger.info("Using refactored processing module structure")
