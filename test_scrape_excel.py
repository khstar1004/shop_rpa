import logging
import os
import sys
from datetime import datetime

# Add project directory to path (if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from core.processing.main_processor import ProductProcessor
from utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_scrape_excel")

def main():
    """Test scraping data and saving to Excel"""
    logger.info("Starting test of data scraping and Excel generation")
    
    # Load configuration
    config = load_config()
    
    # Create product processor
    processor = ProductProcessor(config)
    
    # Define search query
    search_query = "사무용품"  # Office supplies
    
    # Define output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"test_results_{timestamp}.xlsx")
    
    # Process search results
    try:
        logger.info(f"Processing search for: {search_query}")
        processor.process_search_results(search_query, output_file, max_items=10)
        logger.info(f"Test completed successfully. Output file: {output_file}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main() 