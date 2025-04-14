from .reader import ExcelReader
from .formatter import ExcelFormatter
from .writer import ExcelWriter
from .converter import ExcelConverter
from .postprocessor import ExcelPostProcessor
import pandas as pd

class ExcelManager:
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger
        self.reader = ExcelReader(config, logger)
        self.formatter = ExcelFormatter(config, logger)
        self.writer = ExcelWriter(config, logger)
        self.converter = ExcelConverter(config, logger)
        self.postprocessor = ExcelPostProcessor(config, logger)

    def read_excel(self, file_path: str) -> pd.DataFrame:
        return self.reader.read_excel_file(file_path)

    def format_excel(self, file_path: str) -> None:
        self.formatter.apply_formatting_to_excel(file_path)

    def write_excel(self, products: list, output_path: str, sheet_name: str = None, naver_results: list = None) -> str:
        return self.writer.save_products(products, output_path, sheet_name, naver_results)

    def convert_xls_to_xlsx(self, input_directory: str) -> str:
        return self.converter.convert_xls_to_xlsx(input_directory)

    def post_process_excel(self, file_path: str) -> str:
        return self.postprocessor.post_process_excel_file(file_path)

    def add_hyperlinks(self, file_path: str) -> str:
        return self.formatter.add_hyperlinks_to_excel(file_path)

    def filter_by_price_diff(self, file_path: str) -> str:
        return self.formatter.filter_excel_by_price_diff(file_path)

    def remove_at_symbol(self, file_path: str) -> str:
        return self.postprocessor.remove_at_symbol(file_path)

    def save_products(self, products: list, output_path: str, sheet_name: str = None, naver_results: list = None) -> str:
        return self.writer.save_products(products, output_path, sheet_name, naver_results) 