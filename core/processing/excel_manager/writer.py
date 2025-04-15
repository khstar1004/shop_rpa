import logging
import os
import pandas as pd
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter


class ExcelWriter:
    def __init__(self, config: dict, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        if hasattr(config, 'get') and callable(config.get) and hasattr(config, 'sections'):
            try:
                self.excel_settings = dict(config["EXCEL"]) if "EXCEL" in config.sections() else {}
            except Exception as e:
                self.logger.warning(f"Excel 설정을 불러오는 중 오류 발생: {str(e)}")
                self.excel_settings = {}
        else:
            self.excel_settings = config.get("EXCEL", {})

    def generate_enhanced_output(self, results: list, input_file: str, output_dir: str = None) -> str:
        try:
            report_data = []
            processed_count = 0
            for result in results:
                if hasattr(result, "source_product") and result.source_product:
                    processed_count += 1
                    row = {}
                    source_data = getattr(result.source_product, "original_input_data", {})
                    if isinstance(source_data, dict):
                        for key, value in source_data.items():
                            row[key] = value
                    # Ensure basic required fields
                    row.setdefault("상품명", "상품명 누락")
                    row.setdefault("판매단가(V포함)", 0)
                    row.setdefault("상품Code", "DEFAULT-CODE")
                    report_data.append(row)
            if not report_data:
                self.logger.warning("No data to report; creating minimal error row.")
                report_data.append({
                    "상품명": "데이터 처리 중 오류 발생",
                    "판매단가(V포함)": 0,
                    "상품Code": "ERROR",
                    "구분": "ERROR"
                })
            result_df = pd.DataFrame(report_data)
            base_name = os.path.splitext(input_file)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{os.path.basename(base_name)}-result_result_{timestamp}_upload_{timestamp}.xlsx")
            else:
                os.makedirs("output", exist_ok=True)
                output_file = os.path.join("output", f"{os.path.basename(base_name)}-result_result_{timestamp}_upload_{timestamp}.xlsx")
            result_df.to_excel(output_file, index=False)
            self.logger.info(f"Enhanced Excel file saved: {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error generating enhanced output: {str(e)}", exc_info=True)
            return ""

    def create_worksheet(self, workbook: Workbook, sheet_name: str):
        worksheet = workbook.create_sheet(title=sheet_name)
        headers = [
            "제품명", 
            "상품코드", 
            "가격", 
            "메인 이미지", 
            "추가 이미지", 
            "상품 URL", 
            "소스", 
            "브랜드", 
            "카테고리", 
            "스크래핑 시간", 
            "상세 설명",
            "원본 데이터",
            "이미지 URL"
        ]
        for col_idx, header in enumerate(headers, 1):
            cell = worksheet.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        # Set initial column widths
        column_widths = {1:40, 2:15, 3:15, 4:25, 5:25, 6:30, 7:15, 8:20, 9:20, 10:20, 11:40, 12:50, 13:50}
        for col_idx, width in column_widths.items():
            column_letter = get_column_letter(col_idx)
            worksheet.column_dimensions[column_letter].width = width
        worksheet.freeze_panes = "A2"
        return worksheet

    def save_products(self, products: list, output_path: str, sheet_name: str = None, naver_results: list = None) -> str:
        try:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"Created output directory: {output_dir}")
            workbook = Workbook()
            if not sheet_name:
                sheet_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            workbook.remove(workbook.active)
            worksheet = self.create_worksheet(workbook, sheet_name)
            current_row = 2
            if products:
                self.logger.info(f"Writing {len(products)} products to Excel")
                for product in products:
                    current_row = self._write_product_data(worksheet, current_row, product)
            if naver_results:
                self.logger.info(f"Writing {len(naver_results)} Naver results to Excel")
                for product in naver_results:
                    current_row = self._write_product_data(worksheet, current_row, product)
            total_products = (len(products) if products else 0) + (len(naver_results) if naver_results else 0)
            summary_row = current_row + 1
            worksheet.cell(row=summary_row, column=1, value=f"총 제품 수: {total_products}")
            worksheet.cell(row=summary_row, column=2, value=f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            workbook.save(output_path)
            self.logger.info(f"Excel file saved to: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving products to Excel: {str(e)}", exc_info=True)
            return output_path

    def _write_product_data(self, worksheet, row_idx: int, product) -> int:
        try:
            worksheet.cell(row=row_idx, column=1, value=getattr(product, 'name', '제품명 없음'))
            worksheet.cell(row=row_idx, column=2, value=getattr(product, 'product_code', getattr(product, 'id', '코드 없음')))
            worksheet.cell(row=row_idx, column=3, value=getattr(product, 'price', 0))
            image_url = getattr(product, 'image_url', '이미지 없음')
            if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                formula = f'=IMAGE("{image_url}", 2)'
                worksheet.cell(row=row_idx, column=4, value=formula)
            else:
                worksheet.cell(row=row_idx, column=4, value="이미지 없음")
            worksheet.cell(row=row_idx, column=6, value=getattr(product, 'url', 'URL 없음'))
            worksheet.cell(row=row_idx, column=7, value=getattr(product, 'source', '알 수 없음'))
            worksheet.cell(row=row_idx, column=8, value=getattr(product, 'brand', '알 수 없음'))
            worksheet.cell(row=row_idx, column=9, value=getattr(product, 'category', '알 수 없음'))
            worksheet.cell(row=row_idx, column=10, value=getattr(product, 'fetched_at', ''))
            worksheet.cell(row=row_idx, column=11, value=getattr(product, 'description', ''))
            worksheet.cell(row=row_idx, column=12, value=str(getattr(product, 'original_input_data', ''))[:32000])
            return row_idx + 1
        except Exception as e:
            self.logger.error(f"Error writing product data to Excel: {str(e)}", exc_info=True)
            return row_idx + 1 