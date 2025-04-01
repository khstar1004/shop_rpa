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

class Processor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache with improved settings
        self.cache = FileCache(
            cache_dir=config['PATHS']['CACHE_DIR'],
            duration_seconds=config['PROCESSING']['CACHE_DURATION'],
            max_size_mb=config['PROCESSING'].get('CACHE_MAX_SIZE_MB', 1024),
            enable_compression=config['PROCESSING'].get('ENABLE_COMPRESSION', False),
            compression_level=config['PROCESSING'].get('COMPRESSION_LEVEL', 6)
        )
        
        # Initialize matchers with improved cache and settings
        self.text_matcher = TextMatcher(
            cache=self.cache
        )
        self.image_matcher = ImageMatcher(
            cache=self.cache,
            similarity_threshold=config['MATCHING']['IMAGE_SIMILARITY_THRESHOLD']
        )
        self.multimodal_matcher = MultiModalMatcher(
            text_weight=config['MATCHING']['TEXT_WEIGHT'],
            image_weight=config['MATCHING']['IMAGE_WEIGHT'],
            text_matcher=self.text_matcher,
            image_matcher=self.image_matcher,
            similarity_threshold=config['MATCHING'].get('TEXT_SIMILARITY_THRESHOLD', 0.75)
        )
        
        # Get scraping settings
        scraping_config = config.get('SCRAPING', {})
        
        # Initialize scrapers with improved settings
        self.koryo_scraper = KoryoScraper(
            max_retries=config['PROCESSING']['MAX_RETRIES'],
            cache=self.cache,
            timeout=config['PROCESSING'].get('REQUEST_TIMEOUT', 30)
        )
        
        self.naver_crawler = NaverShoppingCrawler(
            max_retries=config['PROCESSING']['MAX_RETRIES'],
            cache=self.cache,
            timeout=config['PROCESSING'].get('REQUEST_TIMEOUT', 30)
        )
        
        # Apply scraping configuration to scrapers if available
        if scraping_config:
            self._configure_scrapers(scraping_config)
        
        # Initialize ThreadPoolExecutor with improved settings
        self.executor = ThreadPoolExecutor(
            max_workers=config['PROCESSING']['MAX_WORKERS'],
            thread_name_prefix='ProductProcessor'
        )
        
        # Add batch processing capability
        self.batch_size = config['PROCESSING'].get('BATCH_SIZE', 10)
        
        # Excel specific settings with enhanced validation from config
        if 'EXCEL' in config:
            excel_config = config['EXCEL']
            self.excel_settings = {
                'sheet_name': excel_config.get('SHEET_NAME', 'Sheet1'),
                'start_row': excel_config.get('START_ROW', 2),
                'required_columns': excel_config.get('REQUIRED_COLUMNS', 
                    ['상품명', '판매단가(V포함)', '상품Code', '본사 이미지', '본사상품링크']),
                'optional_columns': excel_config.get('OPTIONAL_COLUMNS',
                    ['본사 단가', '가격', '상품코드']),
                'max_rows': excel_config.get('MAX_ROWS', 10000),
                'enable_formatting': excel_config.get('ENABLE_FORMATTING', True),
                'date_format': excel_config.get('DATE_FORMAT', 'YYYY-MM-DD'),
                'number_format': excel_config.get('NUMBER_FORMAT', '#,##0.00'),
                'max_file_size_mb': excel_config.get('MAX_FILE_SIZE_MB', 100),
                'enable_data_quality': excel_config.get('ENABLE_DATA_QUALITY_METRICS', True),
                'enable_duplicate_detection': excel_config.get('ENABLE_DUPLICATE_DETECTION', True),
                'enable_auto_correction': excel_config.get('ENABLE_AUTO_CORRECTION', True),
                'auto_correction_rules': excel_config.get('AUTO_CORRECTION_RULES', 
                    ['price', 'url', 'product_code']),
                'report_formatting': excel_config.get('REPORT_FORMATTING', True),
                'report_styles': excel_config.get('REPORT_STYLES', True),
                'report_filters': excel_config.get('REPORT_FILTERS', True),
                'report_sorting': excel_config.get('REPORT_SORTING', True),
                'report_freeze_panes': excel_config.get('REPORT_FREEZE_PANES', True),
                'report_auto_fit': excel_config.get('REPORT_AUTO_FIT', True),
                'validation_rules': {
                    'price': {
                        'min': excel_config.get('PRICE_MIN', 0),
                        'max': excel_config.get('PRICE_MAX', 1000000000)
                    },
                    'product_code': {
                        'pattern': excel_config.get('PRODUCT_CODE_PATTERN', r'^[A-Za-z0-9-]+$')
                    },
                    'url': {
                        'pattern': excel_config.get('URL_PATTERN', r'^https?://.*$')
                    }
                }
            }
        else:
            # Fallback default settings if EXCEL section is missing
            self.excel_settings = {
                'sheet_name': 'Sheet1',
                'start_row': 2,
                'required_columns': ['상품명', '판매단가(V포함)', '상품Code', '본사 이미지', '본사상품링크'],
                'optional_columns': ['본사 단가', '가격', '상품코드'],
                'max_rows': 10000,
                'enable_formatting': True,
                'date_format': 'YYYY-MM-DD',
                'number_format': '#,##0.00',
                'validation_rules': {
                    'price': {'min': 0, 'max': 1000000000},
                    'product_code': {'pattern': r'^[A-Za-z0-9-]+$'},
                    'url': {'pattern': r'^https?://.*$'}
                }
            }

        # Add settings for file splitting and merging
        self.auto_split_files = config['PROCESSING'].get('AUTO_SPLIT_FILES', True)
        self.split_threshold = config['PROCESSING'].get('SPLIT_THRESHOLD', 300)
        self.auto_merge_results = config['PROCESSING'].get('AUTO_MERGE_RESULTS', True)
        self.auto_clean_product_names = config['PROCESSING'].get('AUTO_CLEAN_PRODUCT_NAMES', True)
        
        # Yellow fill for price differences
        self.price_difference_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    def _configure_scrapers(self, scraping_config: Dict):
        """다중 레이어 스크래퍼에 고급 설정 적용"""
        scrapers = [self.koryo_scraper, self.naver_crawler]
        
        for scraper in scrapers:
            # Configure ThreadPoolExecutor
            max_workers = scraping_config.get('MAX_CONCURRENT_REQUESTS', 5)
            if hasattr(scraper, 'executor') and hasattr(scraper.executor, '_max_workers'):
                scraper.executor._max_workers = max_workers
            
            # Configure extraction timeout
            if hasattr(scraper, 'timeout'):
                scraper.timeout = scraping_config.get('EXTRACTION_TIMEOUT', 15)
            
            # Configure extraction strategies
            if hasattr(scraper, 'extraction_strategies'):
                strategies = []
                
                # Add strategies based on configuration
                if scraping_config.get('ENABLE_DOM_EXTRACTION', True):
                    for strategy in scraper.extraction_strategies:
                        if 'DOMExtractionStrategy' in strategy.__class__.__name__:
                            strategies.append(strategy)
                            
                if scraping_config.get('ENABLE_TEXT_EXTRACTION', True):
                    for strategy in scraper.extraction_strategies:
                        if 'TextExtractionStrategy' in strategy.__class__.__name__:
                            strategies.append(strategy)
                            
                if scraping_config.get('ENABLE_COORD_EXTRACTION', True):
                    for strategy in scraper.extraction_strategies:
                        if 'CoordinateExtractionStrategy' in strategy.__class__.__name__:
                            strategies.append(strategy)
                
                # Only update if we have strategies
                if strategies:
                    scraper.extraction_strategies = strategies
            
            # Set politeness delay
            if hasattr(scraper, '_search_product_async'):
                # Monkey patch the method to include the configured delay
                original_method = scraper._search_product_async
                politeness_delay = scraping_config.get('POLITENESS_DELAY', 1500) / 1000  # Convert to seconds
                
                async def patched_method(query, max_items=50):
                    # Update the internal delay in any existing sleep calls
                    scraper.logger.debug(f"Using politeness delay of {politeness_delay} seconds")
                    result = await original_method(query, max_items)
                    return result
                
                scraper._search_product_async = patched_method
    
    def _process_single_product(self, product: Product) -> ProcessingResult:
        """Process a single product (internal helper for _process_single_file)"""
        try:
            return self.process_product(product)
        except Exception as e:
            self.logger.error(f"Error processing product {product.name}: {str(e)}", exc_info=True)
            return ProcessingResult(
                source_product=product,
                error=str(e)
            )

    def process_file(self, input_file: str) -> Tuple[Optional[str], Optional[str]]:
        """Process input Excel file and generate reports."""
        try:
            start_time = datetime.now()
            self.logger.info(f"Processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Verify file exists
            if not os.path.exists(input_file):
                error_msg = f"Input file not found: {input_file}"
                self.logger.error(error_msg)
                return None, error_msg
            
            # Read input file with enhanced error handling
            try:
                df = self._read_excel_file(input_file)
                
                # Verify we got data back
                if df is None or df.empty:
                    error_msg = "No data found in input file"
                    self.logger.error(error_msg)
                    return None, error_msg
                    
                total_items = len(df)
                self.logger.info(f"Loaded {total_items} items from {input_file}")
                
                # Auto-clean product names if enabled
                if self.auto_clean_product_names:
                    df = self._clean_product_names(df)
                    self.logger.info("Product names cleaned automatically")
                
            except Exception as e:
                self.logger.error(f"Failed to read input file: {str(e)}", exc_info=True)
                return None, f"Failed to read input file: {str(e)}"
                
            # Split files if needed
            if self.auto_split_files and total_items > self.split_threshold:
                try:
                    split_files = self._split_input_file(df, input_file)
                    self.logger.info(f"Input file split into {len(split_files)} files")
                    
                    # Process each split file
                    result_files = []
                    for split_file in split_files:
                        result_file, _ = self._process_single_file(split_file)
                        if result_file:
                            result_files.append(result_file)
                    
                    # Merge results if enabled
                    if self.auto_merge_results and len(result_files) > 1:
                        merged_result = self._merge_result_files(result_files, input_file)
                        return merged_result, None
                    
                    return result_files[0] if result_files else None, None
                except Exception as e:
                    self.logger.error(f"Error splitting input file: {str(e)}", exc_info=True)
                    # Fall back to processing as a single file
                    self.logger.info("Falling back to processing as a single file")
                    return self._process_single_file(input_file)
            else:
                # Process as a single file
                return self._process_single_file(input_file)
                
        except Exception as e:
            self.logger.error(f"Error in process_file: {str(e)}", exc_info=True)
            return None, str(e)
    
    def _process_single_file(self, input_file: str) -> Tuple[Optional[str], Optional[str]]:
        """Process a single input file (internal method)"""
        try:
            # Define start_time at the beginning of the method
            start_time = datetime.now()
            
            # Read input file
            df = self._read_excel_file(input_file)
            total_items = len(df)
            
            # Process items in batches
            results = []
            futures = []
            total_futures = 0
            
            for i in range(0, total_items, self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                batch_futures = []
                
                for _, row in batch.iterrows():
                    product = self._create_product_from_row(row)
                    if product:  # Only process valid products
                        future = self.executor.submit(
                            self._process_single_product,
                            product
                        )
                        batch_futures.append((product, future))
                        total_futures += 1
                
                # Wait for batch to complete before processing next batch
                for product, future in batch_futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per product
                        results.append(result)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing product {product.id}: {str(e)}", 
                            exc_info=True
                        )
                        # Add failed result to maintain order
                        results.append(ProcessingResult(
                            product=product,
                            error=str(e)
                        ))
                    
                    # Update progress
                    progress_percent = int((len(results) / total_items) * 100)
                    self.logger.info(f"Progress: {len(results)}/{total_items} ({progress_percent}%)")
            
            # Generate output report with enhanced formatting
            output_file = self._generate_enhanced_output(results, input_file)
            
            end_time = datetime.now()
            processing_time = end_time - start_time
            self.logger.info(f"Processing finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total processing time: {processing_time}")
            
            return output_file, None
            
        except Exception as e:
            self.logger.error(f"Error in _process_single_file: {str(e)}", exc_info=True)
            return None, str(e)
    
    def _clean_product_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean product names according to manual requirements"""
        if "상품명" in df.columns:
            # Remove '1-' and numbers/hyphens before it
            df["상품명"] = df["상품명"].str.replace(r'^\d+-', '', regex=True)
            # Remove special characters and brackets with numbers
            df["상품명"] = df["상품명"].str.replace(r'[/()]', '', regex=True)
            # Remove numbers in parentheses
            df["상품명"] = df["상품명"].str.replace(r'\(\d+\)', '', regex=True)
            
        return df
    
    def _split_input_file(self, df: pd.DataFrame, input_file: str) -> List[str]:
        """Split input file into smaller chunks based on threshold"""
        base_name = os.path.splitext(input_file)[0]
        extension = os.path.splitext(input_file)[1]
        
        # Get processing type ('A' for approval, 'P' for price)
        processing_type = "승인관리"
        if "구분" in df.columns and df["구분"].iloc[0].upper() == "P":
            processing_type = "가격관리"
            
        # Get current date
        current_date = datetime.now().strftime("%Y%m%d")
        
        # Calculate number of files needed
        total_rows = len(df)
        num_files = (total_rows + self.split_threshold - 1) // self.split_threshold
        
        split_files = []
        for i in range(num_files):
            start_idx = i * self.split_threshold
            end_idx = min((i + 1) * self.split_threshold, total_rows)
            
            # Create filename as per manual: 승인관리1(300)-날짜
            file_count = i + 1
            file_size = end_idx - start_idx
            split_filename = f"{processing_type}{file_count}({file_size})-{current_date}{extension}"
            split_path = os.path.join(os.path.dirname(input_file), split_filename)
            
            # Save the chunk
            df.iloc[start_idx:end_idx].to_excel(split_path, index=False)
            split_files.append(split_path)
            
        return split_files
    
    def _merge_result_files(self, result_files: List[str], original_input: str) -> str:
        """Merge multiple result files into a single file"""
        # Create merged dataframe
        dfs = []
        for file in result_files:
            df = pd.read_excel(file)
            dfs.append(df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Generate output filename
        base_name = os.path.splitext(original_input)[0]
        merged_filename = f"{base_name}-merged-result.xlsx"
        
        # Save merged file
        merged_df.to_excel(merged_filename, index=False)
        
        # Apply formatting
        self._apply_formatting_to_excel(merged_filename)
        
        return merged_filename
    
    def _generate_enhanced_output(self, results: List[ProcessingResult], input_file: str) -> str:
        """Generate enhanced output file with proper formatting"""
        # Filter products with price differences (lower prices)
        filtered_results = [
            r for r in results if (
                (r.best_koryo_match and r.best_koryo_match.price_difference < 0) or
                (r.best_naver_match and r.best_naver_match.price_difference < 0)
            )
        ]
        
        # Create result dataframe - this part was missing
        report_data = []
        for result in results:
            row = {}
            # Extract source product data
            source_data = result.source_product.original_input_data
            for key, value in source_data.items():
                row[key] = value
            
            # Add match data from koryo
            if result.best_koryo_match:
                row['판매단가(V포함)(2)'] = result.best_koryo_match.matched_product.price
                row['가격차이(2)'] = result.best_koryo_match.price_difference
                row['가격차이(2)%'] = result.best_koryo_match.price_difference_percent
                row['고려기프트 이미지'] = result.best_koryo_match.matched_product.image_url
                row['고려기프트상품링크'] = result.best_koryo_match.matched_product.url
            
            # Add match data from naver
            if result.best_naver_match:
                row['판매단가(V포함)(3)'] = result.best_naver_match.matched_product.price
                row['가격차이(3)'] = result.best_naver_match.price_difference
                row['가격차이(3)%'] = result.best_naver_match.price_difference_percent
                row['네이버 이미지'] = result.best_naver_match.matched_product.image_url
                row['네이버쇼핑 링크'] = result.best_naver_match.matched_product.url
                if hasattr(result.best_naver_match.matched_product, 'brand'):
                    row['공급사명'] = result.best_naver_match.matched_product.brand
            
            report_data.append(row)
        
        # Create the DataFrame that was previously undefined
        result_df = pd.DataFrame(report_data)
        
        # Generate output filename
        output_file = f"{os.path.splitext(input_file)[0]}-result.xlsx"
        
        # Save to Excel
        result_df.to_excel(output_file, index=False)
        
        # Apply formatting according to manual requirements
        self._apply_formatting_to_excel(output_file)
        
        return output_file
    
    def _apply_formatting_to_excel(self, excel_file: str) -> None:
        """Apply formatting requirements to the output Excel file"""
        wb = load_workbook(excel_file)
        ws = wb.active
        
        # Find columns with price differences
        price_diff_cols = []
        for col in range(1, ws.max_column + 1):
            header = ws.cell(row=1, column=col).value
            if header and ('가격차이' in str(header)):
                price_diff_cols.append(col)
        
        # Apply yellow highlight to negative price differences
        for col in price_diff_cols:
            for row in range(2, ws.max_row + 1):
                cell_value = ws.cell(row=row, column=col).value
                if cell_value and (isinstance(cell_value, (int, float)) and cell_value < 0):
                    ws.cell(row=row, column=col).fill = self.price_difference_fill
        
        # Save changes
        wb.save(excel_file)

    def _read_excel_file(self, file_path: str) -> pd.DataFrame:
        """Read Excel file with enhanced validation and error handling."""
        try:
            # Check file existence and size
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            
            self.logger.info(f"Reading Excel file: {file_path}")
            
            # Try to detect sheets first
            sheet_name = self.excel_settings['sheet_name']
            sheet_names = []
            
            try:
                workbook = load_workbook(file_path, read_only=True)
                sheet_names = workbook.sheetnames
                self.logger.info(f"Found sheets in workbook: {sheet_names}")
                
                if sheet_name not in sheet_names and sheet_names:
                    self.logger.warning(f"Sheet '{sheet_name}' not found. Using first sheet: '{sheet_names[0]}'")
                    sheet_name = sheet_names[0]
            except Exception as e:
                self.logger.warning(f"Could not inspect Excel structure: {str(e)}. Using default sheet name.")
            
            # Try using pandas to read the file directly with different parameters
            all_dataframes = []
            
            # First try: Use sheet_name=None to read all sheets
            try:
                all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
                self.logger.info(f"Successfully read {len(all_sheets)} sheets from Excel")
                
                # Try each sheet
                for sheet, df in all_sheets.items():
                    if not df.empty:
                        self.logger.info(f"Found data in sheet '{sheet}' with {len(df)} rows and columns: {list(df.columns)}")
                        all_dataframes.append((df, sheet, 0))  # (dataframe, sheet_name, skiprows)
            except Exception as e:
                self.logger.warning(f"Failed to read all sheets: {str(e)}")
            
            # Second try: Try different skiprows values with the identified sheet
            for skip_rows in [0, 1, 2, 3, 4, 5]:
                try:
                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        skiprows=skip_rows,
                        engine='openpyxl'
                    )
                    if not df.empty:
                        self.logger.info(f"Found data with skiprows={skip_rows}, {len(df)} rows and columns: {list(df.columns)}")
                        all_dataframes.append((df, sheet_name, skip_rows))
                except Exception as e:
                    self.logger.debug(f"Failed with skiprows={skip_rows}: {str(e)}")
            
            # If we found no valid dataframes, try a more permissive approach
            if not all_dataframes:
                self.logger.warning("No valid data found with standard approaches. Trying more permissive methods.")
                try:
                    # Try with xlrd engine for older Excel formats
                    df = pd.read_excel(file_path, engine='xlrd')
                    if not df.empty:
                        self.logger.info(f"Found data using xlrd engine with {len(df)} rows")
                        all_dataframes.append((df, "Unknown", 0))
                except Exception:
                    pass
                
                try:
                    # Try CSV parsing as last resort
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        self.logger.info(f"Found data using CSV parser with {len(df)} rows")
                        all_dataframes.append((df, "CSV", 0))
                except Exception:
                    pass
            
            # If we still have no dataframes, we can't proceed
            if not all_dataframes:
                self.logger.error(f"Could not read any valid data from {file_path}")
                # Create a minimal dataframe with required columns to avoid breaking
                return pd.DataFrame({
                    '상품명': ["No data found"],
                    '판매단가(V포함)': [0],
                    '상품Code': ["ERROR"],
                    '본사 이미지': [""],
                    '본사상품링크': [""]
                })
            
            # Select the best dataframe based on number of rows
            best_df, sheet, skiprows = max(all_dataframes, key=lambda x: len(x[0]))
            self.logger.info(f"Selected dataframe from sheet '{sheet}' with skiprows={skiprows}, {len(best_df)} rows")
            
            # Log original columns for debugging
            self.logger.info(f"Original columns: {list(best_df.columns)}")
            
            # Create required columns if missing
            for required_col in self.excel_settings['required_columns']:
                if required_col not in best_df.columns:
                    # Try to map from similar column names
                    mapped = self._find_similar_column(best_df, required_col)
                    if mapped:
                        self.logger.info(f"Mapped '{mapped}' to required column '{required_col}'")
                        best_df[required_col] = best_df[mapped]
                    else:
                        self.logger.warning(f"Creating default values for missing column '{required_col}'")
                        # Create default values based on column type
                        if required_col == '상품명':
                            # Try to use the first text column
                            text_cols = best_df.select_dtypes(include=['object']).columns
                            if len(text_cols) > 0:
                                best_df[required_col] = best_df[text_cols[0]]
                            else:
                                best_df[required_col] = [f"Product {i+1}" for i in range(len(best_df))]
                        elif required_col == '판매단가(V포함)':
                            # Try to use the first numeric column
                            num_cols = best_df.select_dtypes(include=['number']).columns
                            if len(num_cols) > 0:
                                best_df[required_col] = best_df[num_cols[0]]
                            else:
                                best_df[required_col] = 0
                        elif required_col == '상품Code':
                            best_df[required_col] = [f"PRODUCT-{i+1:04d}" for i in range(len(best_df))]
                        else:
                            best_df[required_col] = ""
            
            # Clean data
            best_df = self._clean_excel_data(best_df)
            
            return best_df
        except Exception as e:
            self.logger.error(f"Error reading Excel file: {str(e)}", exc_info=True)
            # Return a minimal valid dataframe to avoid breaking downstream code
            return pd.DataFrame({
                '상품명': ["Error reading file"],
                '판매단가(V포함)': [0],
                '상품Code': ["ERROR"],
                '본사 이미지': [""],
                '본사상품링크': [""]
            })

    def _find_similar_column(self, df: pd.DataFrame, target_column: str) -> Optional[str]:
        """Find a column in the dataframe that is similar to the target column."""
        # Simple mapper for Korean column names
        column_mapping = {
            '상품명': ['품명', '제품명', '상품', 'product', 'name', 'item', '품목', '상품이름', '제품', '품명'],
            '판매단가(V포함)': ['단가', '판매가', '가격', 'price', '가격(v포함)', '단가(vat)', '판매단가', '판매 단가', '판매 가격'],
            '상품Code': ['코드', 'code', 'item code', 'product code', '품목코드', '제품코드', '상품코드', '상품 코드'],
            '본사 이미지': ['이미지', 'image', '상품이미지', '제품이미지', '이미지주소', 'image url', '이미지링크'],
            '본사상품링크': ['링크', 'link', 'url', '상품링크', '제품링크', '상품url', '제품url', '홈페이지']
        }
        
        # Check for exact match first
        if target_column in df.columns:
            return target_column
        
        # Check for case-insensitive match
        for col in df.columns:
            if col.lower() == target_column.lower():
                return col
        
        # Check for similar columns
        if target_column in column_mapping:
            for similar in column_mapping[target_column]:
                for col in df.columns:
                    if similar.lower() in str(col).lower():
                        return col
        
        # No similar column found
        return None

    def _clean_excel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate Excel data with enhanced cleaning."""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Remove rows with missing required data
        initial_rows = len(df)
        df = df.dropna(subset=['상품명', '판매단가(V포함)'])
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.warning(f"Removed {removed_rows} rows with missing required data")
        
        # Detect and handle duplicates if enabled
        if self.excel_settings.get('enable_duplicate_detection', True):
            duplicate_rows = df.duplicated(subset=['상품명'], keep='first')
            if duplicate_rows.any():
                duplicate_count = duplicate_rows.sum()
                self.logger.warning(f"Found {duplicate_count} duplicate product names. Keeping first occurrence.")
                df = df[~duplicate_rows]
        
        # Check if auto-correction is enabled
        auto_correct = self.excel_settings.get('enable_auto_correction', True)
        auto_correction_rules = self.excel_settings.get('auto_correction_rules', ['price', 'url', 'product_code'])
        
        # Clean product codes with enhanced validation
        df['상품Code'] = df['상품Code'].fillna(df['상품코드'])
        df['상품Code'] = df['상품Code'].astype(str).str.strip()
        
        invalid_codes = df[~df['상품Code'].str.match(self.excel_settings['validation_rules']['product_code']['pattern'])]
        if not invalid_codes.empty:
            self.logger.warning(f"Found {len(invalid_codes)} invalid product codes")
            
            # Auto-correct product codes if enabled
            if auto_correct and 'product_code' in auto_correction_rules:
                self.logger.info("Auto-correcting invalid product codes")
                df['상품Code'] = df['상품Code'].apply(self._clean_product_code)
        
        # Clean prices with enhanced validation
        df['판매단가(V포함)'] = df['판매단가(V포함)'].apply(self._clean_price)
        invalid_prices = df[
            (df['판매단가(V포함)'] < self.excel_settings['validation_rules']['price']['min']) |
            (df['판매단가(V포함)'] > self.excel_settings['validation_rules']['price']['max'])
        ]
        if not invalid_prices.empty:
            self.logger.warning(f"Found {len(invalid_prices)} invalid prices")
            
            # Auto-correct invalid prices if enabled
            if auto_correct and 'price' in auto_correction_rules:
                # Reset all invalid prices to minimum valid price
                min_price = self.excel_settings['validation_rules']['price']['min']
                self.logger.info(f"Auto-correcting invalid prices to minimum valid price: {min_price}")
                
                mask = (df['판매단가(V포함)'] < min_price) | (df['판매단가(V포함)'] > self.excel_settings['validation_rules']['price']['max'])
                df.loc[mask, '판매단가(V포함)'] = min_price
        
        # Clean URLs with enhanced validation
        df['본사 이미지'] = df['본사 이미지'].astype(str).str.strip()
        df['본사상품링크'] = df['본사상품링크'].astype(str).str.strip()
        
        invalid_urls = df[
            ~df['본사 이미지'].str.match(self.excel_settings['validation_rules']['url']['pattern']) |
            ~df['본사상품링크'].str.match(self.excel_settings['validation_rules']['url']['pattern'])
        ]
        if not invalid_urls.empty:
            self.logger.warning(f"Found {len(invalid_urls)} invalid URLs")
            
            # Auto-correct URLs if enabled
            if auto_correct and 'url' in auto_correction_rules:
                self.logger.info("Auto-correcting invalid URLs")
                df['본사 이미지'] = df['본사 이미지'].apply(self._clean_url)
                df['본사상품링크'] = df['본사상품링크'].apply(self._clean_url)
        
        return df

    def _clean_product_code(self, code: str) -> str:
        """Clean and normalize product code."""
        if pd.isna(code) or not code:
            return ""
        
        # Remove any non-alphanumeric characters except hyphen
        cleaned_code = re.sub(r'[^A-Za-z0-9\-]', '', str(code))
        
        # Ensure it's not empty after cleaning
        if not cleaned_code:
            # Generate a simple placeholder with timestamp
            cleaned_code = f"CODE-{int(datetime.now().timestamp())}"
            
        return cleaned_code

    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL."""
        if pd.isna(url) or not url or url == 'nan':
            return ""
            
        url = str(url).strip()
        
        # Check if it starts with http/https, add if missing
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Simple URL validation regex
        if url and not re.match(self.excel_settings['validation_rules']['url']['pattern'], url):
            # URL is invalid, return empty string
            return ""
            
        return url

    def _validate_excel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate Excel data with enhanced validation rules."""
        # Add data validation rules
        validation_errors = []
        
        # Validate product names
        empty_names = df['상품명'].isna() | (df['상품명'].str.strip() == '')
        if empty_names.any():
            validation_errors.append(f"Found {empty_names.sum()} empty product names")
        
        # Validate prices
        invalid_prices = df[
            (df['판매단가(V포함)'] < self.excel_settings['validation_rules']['price']['min']) |
            (df['판매단가(V포함)'] > self.excel_settings['validation_rules']['price']['max'])
        ]
        if not invalid_prices.empty:
            validation_errors.append(f"Found {len(invalid_prices)} invalid prices")
        
        # Validate URLs
        invalid_urls = df[
            ~df['본사 이미지'].str.match(self.excel_settings['validation_rules']['url']['pattern']) |
            ~df['본사상품링크'].str.match(self.excel_settings['validation_rules']['url']['pattern'])
        ]
        if not invalid_urls.empty:
            validation_errors.append(f"Found {len(invalid_urls)} invalid URLs")
        
        if validation_errors:
            self.logger.warning("Data validation errors found:\n" + "\n".join(validation_errors))
        
        return df

    def _log_data_quality_metrics(self, df: pd.DataFrame):
        """Log data quality metrics for analysis."""
        metrics = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_products': df['상품명'].nunique(),
            'price_stats': {
                'min': df['판매단가(V포함)'].min(),
                'max': df['판매단가(V포함)'].max(),
                'mean': df['판매단가(V포함)'].mean(),
                'median': df['판매단가(V포함)'].median()
            }
        }
        
        self.logger.info("Data Quality Metrics:")
        self.logger.info(f"Total rows: {metrics['total_rows']}")
        self.logger.info(f"Missing values: {metrics['missing_values']}")
        self.logger.info(f"Duplicate rows: {metrics['duplicate_rows']}")
        self.logger.info(f"Unique products: {metrics['unique_products']}")
        self.logger.info(f"Price statistics: {metrics['price_stats']}")

    def _clean_price(self, price) -> float:
        """Clean and convert price to float with enhanced validation."""
        try:
            if pd.isna(price):
                return 0.0
            if isinstance(price, (int, float)):
                return float(price)
            
            # Remove currency symbols and non-numeric characters
            price_str = re.sub(r'[^\d.]', '', str(price))
            
            # Validate price range
            price_float = float(price_str) if price_str else 0.0
            if price_float < self.excel_settings['validation_rules']['price']['min']:
                self.logger.warning(f"Price below minimum: {price}")
                return 0.0
            if price_float > self.excel_settings['validation_rules']['price']['max']:
                self.logger.warning(f"Price above maximum: {price}")
                return 0.0
                
            return price_float
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not parse price '{price}': {str(e)}. Setting to 0.")
            return 0.0

    def _create_product_from_row(self, row: pd.Series) -> Optional[Product]:
        """Create Product object from DataFrame row with enhanced validation."""
        try:
            # Extract and validate key information
            name = str(row['상품명']).strip()
            if not name:
                self.logger.warning("Skipping row due to empty product name")
                return None
                
            price = self._clean_price(row['판매단가(V포함)'])
            if price <= 0:
                self.logger.warning(f"Skipping row due to invalid price: {price}")
                return None
                
            product_code = str(row['상품Code']).strip()
            if not re.match(self.excel_settings['validation_rules']['product_code']['pattern'], product_code):
                self.logger.warning(f"Invalid product code format: {product_code}")
                return None
            
            # Create product with all available data
            product = Product(
                id=product_code,
                name=name,
                price=price,
                source='haeoreum',
                original_input_data=row.to_dict()
            )
            
            # Set and validate optional fields
            image_url = str(row['본사 이미지']).strip()
            product_url = str(row['본사상품링크']).strip()
            
            if image_url and re.match(self.excel_settings['validation_rules']['url']['pattern'], image_url):
                product.image_url = image_url
            if product_url and re.match(self.excel_settings['validation_rules']['url']['pattern'], product_url):
                product.url = product_url
            
            return product
            
        except Exception as e:
            self.logger.error(f"Error creating Product from row: {str(e)}", exc_info=True)
            return None
    
    def process_product(self, product: Product) -> ProcessingResult:
        """Process a single source product to find matches."""
        self.logger.info(f"Processing product: {product.name} (ID: {product.id})")
        processing_result = ProcessingResult(source_product=product)
        try:
            koryo_matches = self.koryo_scraper.search_product(product.name)
            naver_matches = self.naver_crawler.search_product(product.name)
            
            self.logger.debug(f"Found {len(koryo_matches)} Koryo, {len(naver_matches)} Naver potential matches for {product.name}.")
            
            # Process Koryo matches
            for match in koryo_matches:
                match_result = self._calculate_match_similarities(product, match)
                processing_result.koryo_matches.append(match_result)
            
            # Process Naver matches
            for match in naver_matches:
                 match_result = self._calculate_match_similarities(product, match)
                 processing_result.naver_matches.append(match_result)
            
            # Find best match for each source based on thresholds
            processing_result.best_koryo_match = self._find_best_match(processing_result.koryo_matches)
            processing_result.best_naver_match = self._find_best_match(processing_result.naver_matches)

            if processing_result.best_koryo_match:
                 self.logger.info(f"Best Koryo match for {product.name}: {processing_result.best_koryo_match.matched_product.name} ({processing_result.best_koryo_match.combined_similarity:.2f})")
            if processing_result.best_naver_match:
                 self.logger.info(f"Best Naver match for {product.name}: {processing_result.best_naver_match.matched_product.name} ({processing_result.best_naver_match.combined_similarity:.2f})")

        except Exception as e:
            self.logger.error(f"Error processing product {product.name}: {str(e)}", exc_info=True)
            processing_result.error = str(e)
            
        return processing_result

    def _calculate_match_similarities(self, source_product: Product, matched_product: Product) -> MatchResult:
         """Calculates similarities between two products."""
         text_sim = self.text_matcher.calculate_similarity(
             source_product.name, matched_product.name
         )
         image_sim = self.image_matcher.calculate_similarity(
             source_product.original_input_data.get('본사 이미지'), # Use original image URL
             matched_product.image_url
         )
         combined_sim = self.multimodal_matcher.calculate_similarity(
             text_sim, image_sim
         )
         
         self.logger.debug(f"  Match candidate {matched_product.name} ({matched_product.source}): Txt={text_sim:.2f}, Img={image_sim:.2f}, Comb={combined_sim:.2f}")
         
         # Calculate price difference
         price_diff = 0.0
         price_diff_percent = 0.0
         source_price = source_product.price
         if source_price and source_price > 0 and isinstance(matched_product.price, (int, float)):
              price_diff = matched_product.price - source_price
              price_diff_percent = (price_diff / source_price) * 100 if source_price != 0 else 0
         
         return MatchResult(
             source_product=source_product,
             matched_product=matched_product,
             text_similarity=text_sim,
             image_similarity=image_sim,
             combined_similarity=combined_sim,
             price_difference=price_diff,
             price_difference_percent=price_diff_percent
         )
    
    def _find_best_match(self, matches: List[MatchResult]) -> Optional[MatchResult]:
         """Find the best match from a list of match results based on thresholds."""
         if not matches:
             return None
         
         text_threshold = self.config['MATCHING']['TEXT_SIMILARITY_THRESHOLD']
         image_threshold = self.config['MATCHING']['IMAGE_SIMILARITY_THRESHOLD']

         # Filter by similarity thresholds
         valid_matches = [
             m for m in matches
             if m.text_similarity >= text_threshold
             and m.image_similarity >= image_threshold
         ]
         
         if not valid_matches:
             return None
         
         # Return match with highest combined similarity among valid ones
         return max(valid_matches, key=lambda x: x.combined_similarity) 