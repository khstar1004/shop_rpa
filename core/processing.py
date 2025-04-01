import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import re
import os
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
import numpy as np
from pathlib import Path

from .data_models import Product, MatchResult, ProcessingResult
from .matching.text_matcher import TextMatcher
from .matching.image_matcher import ImageMatcher
from .matching.multimodal_matcher import MultiModalMatcher
from .scraping.koryo_scraper import KoryoScraper
from .scraping.naver_crawler import NaverShoppingCrawler
from utils.reporting import generate_primary_report, generate_secondary_report
from utils.caching import FileCache, cache_result

# Import refactored module structure while maintaining backward compatibility
import logging
from typing import Dict, Optional, Tuple, List
import os
from pathlib import Path
import warnings

# Import the refactored Processor class
from .processing.main_processor import ProductProcessor as Processor
from .data_models import Product, MatchResult, ProcessingResult

# For backward compatibility, re-export these classes
__all__ = ['Processor', 'Product', 'MatchResult', 'ProcessingResult']

# Show deprecation warning
warnings.warn(
    "Direct import from 'processing' is deprecated. Use the new module structure instead: "
    "from core.processing import Processor",
    DeprecationWarning,
    stacklevel=2
)

# Setup module logger
logger = logging.getLogger(__name__)
logger.info("Using refactored processing module structure")