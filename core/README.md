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

- **Text Matching**: Advanced Korean text similarity with Levenshtein and BERT embeddings
- **Image Matching**: Background removal and feature comparison using ImageHash and EfficientNet
- **Multimodal Matching**: Combined text/image matching scores with configurable weights 
- **Price Analysis**: Automatic calculation of price differences and percentage discounts

## Workflow Integration

This system integrates with the manual workflow described in 작업메뉴얼.txt:
1. **승인관리/가격관리** identification via the '구분' column (A/P values)
2. Automatic handling of the 300-product file split requirement
3. Output formatting per specifications for price-differing products

## Dependencies

- Pandas and OpenpyXL for Excel processing
- Sentence-Transformers for text similarity
- EfficientNet and ImageHash for image matching
- Rembg for product image background removal

## Usage

The main entry point is the `Processor` class which handles all file operations, matching logic, and report generation.

```python
from core.processing import Processor
processor = Processor(config)
result_file, error = processor.process_file("input.xlsx")
``` 