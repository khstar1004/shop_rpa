# Shop RPA Core Module

The core processing engine for the Shop RPA system, responsible for comparing product prices across Haeoreum Gift, Koryo Gift, and Naver Shopping.

## New Features

### Automatic File Management
- **Auto File Splitting**: Files with more than 300 products are automatically split according to manual workflow requirements
- **Auto File Merging**: Results are automatically merged back together after processing
- **Product Name Cleaning**: Automatically removes prefixes like '1-' and special characters as per manual requirements
- **Enhanced Excel Formatting**: Applies yellow highlighting to price differences

### Configuration Options
```yaml
PROCESSING:
  AUTO_SPLIT_FILES: true       # Enable/disable automatic file splitting
  SPLIT_THRESHOLD: 300         # Threshold for splitting files
  AUTO_MERGE_RESULTS: true     # Enable/disable automatic result merging
  AUTO_CLEAN_PRODUCT_NAMES: true  # Enable/disable product name cleaning
```

## Main Components

- **`scraping/`**: Contains modules for scraping product information from different online stores (Haeoreum Gift, Koryo Gift, Naver Shopping). Uses tools like Playwright and Selenium to extract details like price tables, stock status, and product search results.
- **`processing/`**: Handles the processing of scraped data. This includes cleaning data, managing Excel files (splitting large files, merging results, applying formatting), and transforming data structures. It likely orchestrates the overall data flow.
- **`matching/`**: Implements the logic for matching products based on similarity. It uses techniques for:
    - **Text Matching**: Advanced Korean text similarity using algorithms like Levenshtein distance and potentially language models like BERT embeddings.
    - **Image Matching**: Compares product images by removing backgrounds (using libraries like `rembg`) and comparing visual features (using libraries like `ImageHash` and models like `EfficientNet`).
    - **Multimodal Matching**: Combines text and image matching scores, possibly with configurable weights, to determine the best matches.
- **`data_models.py`**: Defines the core data structures (like `Product`, `MatchResult`, etc.) used throughout the `core` module and potentially other parts of the application to ensure consistent data handling.
- **Price Analysis**: Includes logic for automatic calculation of price differences and percentage discounts between matched products.

## Workflow Integration

This system integrates with the manual workflow described in 작업메뉴얼.txt:
1. **승인관리/가격관리** identification via the '구분' column (A/P values)
2. Automatic handling of the 300-product file split requirement
3. Output formatting per specifications for price-differing products

## Dependencies

- Pandas and Openpyxl for Excel processing
- Sentence-Transformers for text similarity
- EfficientNet and ImageHash for image matching
- Rembg for product image background removal
- Playwright/Selenium for web scraping

## Usage

The main entry point is likely a class within the `processing` module (e.g., `Processor` class mentioned) which handles file operations, matching logic, and report generation.

```python
# Example (assuming Processor class exists in core.processing)
# from core.processing import Processor
# processor = Processor(config)
# result_file, error = processor.process_file("input.xlsx")
``` 