import os
import unittest
import pandas as pd
import shutil
import logging
from datetime import datetime
from pathlib import Path

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in os.sys.path:
    os.sys.path.append(ROOT_DIR)

# 필요한 클래스 임포트
from core.processing.excel_manager import ExcelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestExcelFullProcess(unittest.TestCase):
    """Excel 처리 전체 과정 테스트"""

    @classmethod
    def setUpClass(cls):
        """테스트 시작 전 설정"""
        cls.test_dir = os.path.join(CURRENT_DIR, "test_excel_full")
        cls.output_dir = os.path.join(cls.test_dir, "output")
        
        # 테스트 디렉토리 생성
        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # 테스트용 설정
        cls.config = {
            "EXCEL": {
                "sheet_name": "Test",
                "required_columns": [
                    "상품명",
                    "판매단가(V포함)",
                    "상품Code",
                    "본사 이미지",
                    "본사상품링크",
                ],
            }
        }
        
        # Excel Manager 인스턴스 생성
        cls.excel_manager = ExcelManager(cls.config, logger=logger)
        
        logger.info(f"Test environment set up in: {cls.test_dir}")
        
    @classmethod
    def tearDownClass(cls):
        """테스트 완료 후 정리"""
        try:
            shutil.rmtree(cls.test_dir)
            logger.info(f"Removed test directory: {cls.test_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up test directory: {e}")
            
    def setUp(self):
        """각 테스트 시작 전 설정"""
        # 테스트 데이터 초기화
        self.test_data = pd.DataFrame({
            "상품명": ["테스트 상품 A", "테스트 상품 B", "테스트 상품 C"],
            "판매단가(V포함)": [10000, 20000, 30000],
            "상품Code": ["TEST-001", "TEST-002", "TEST-003"],
            "본사 이미지": [
                "https://example.com/image1.jpg",
                "https://user@example.com/image2.jpg",  # @ 기호 포함
                "https://example.com/image3.jpg",
            ],
            "본사상품링크": [
                "https://example.com/product1",
                "https://example.com/product2",
                "https://user@example.com/product3",  # @ 기호 포함
            ]
        })
        
        # 테스트 파일 경로
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_file = os.path.join(self.test_dir, f"test_input_{self.timestamp}.xlsx")
        
        # 테스트 데이터 저장
        self.test_data.to_excel(self.test_file, index=False)
        logger.info(f"Created test Excel file: {self.test_file}")
        
    def test_excel_file_reading(self):
        """Excel 파일 읽기 테스트"""
        # ExcelManager의 read_excel_file 메서드 테스트
        df = self.excel_manager.read_excel_file(self.test_file)
        
        # 데이터프레임 유효성 검증
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertEqual(len(df), 3, "DataFrame should have 3 rows")
        self.assertIn("상품명", df.columns, "DataFrame should have '상품명' column")
        self.assertIn("판매단가(V포함)", df.columns, "DataFrame should have '판매단가(V포함)' column")
        
        logger.info("Excel file reading test passed")
        
    def test_excel_formatting(self):
        """Excel 포맷팅 테스트"""
        # 포맷팅 적용
        try:
            self.excel_manager.apply_formatting_to_excel(self.test_file)
            
            # 포맷팅 후 파일 크기 확인 (증가했을 것으로 예상)
            file_size_after = os.path.getsize(self.test_file)
            self.assertGreater(file_size_after, 0, "Formatted file should not be empty")
            
            logger.info(f"Formatted Excel file size: {file_size_after} bytes")
            logger.info("Excel formatting test passed")
        except Exception as e:
            self.fail(f"Excel formatting failed: {str(e)}")
            
    def test_excel_post_processing(self):
        """Excel 후처리 테스트"""
        # 후처리 테스트
        try:
            # 원본 파일 크기 기록
            original_size = os.path.getsize(self.test_file)
            
            # 후처리 적용
            processed_file = self.excel_manager.post_process_excel_file(self.test_file)
            
            # 후처리 파일 존재 확인
            self.assertTrue(os.path.exists(processed_file), "Processed file should exist")
            
            # 파일 크기 확인
            processed_size = os.path.getsize(processed_file)
            logger.info(f"Original file size: {original_size}, Processed file size: {processed_size}")
            
            # 후처리 파일에서 @ 기호 확인
            # at_sign_removed = True
            # if processed_file == self.test_file:  # 같은 파일인 경우
            #     df = pd.read_excel(processed_file)
            #     for col in ["본사 이미지", "본사상품링크"]:
            #         if col in df.columns:
            #             for value in df[col]:
            #                 if '@' in str(value):
            #                     at_sign_removed = False
            #                     break
            # self.assertTrue(at_sign_removed, "@ signs should be removed in URLs")
            
            logger.info("Excel post-processing test passed")
        except Exception as e:
            self.fail(f"Excel post-processing failed: {str(e)}")
            
    def test_remove_at_symbol(self):
        """@ 기호 제거 테스트"""
        try:
            # @ 기호 제거 적용
            cleaned_file = self.excel_manager.remove_at_symbol(self.test_file)
            
            # 파일 존재 확인
            self.assertTrue(os.path.exists(cleaned_file), "Cleaned file should exist")
            
            # 파일 내용에서 @ 기호 확인
            df = pd.read_excel(cleaned_file)
            at_sign_found = False
            
            for col in ["본사 이미지", "본사상품링크"]:
                if col in df.columns:
                    for value in df[col].astype(str):
                        if '@' in value:
                            at_sign_found = True
                            logger.warning(f"Found @ sign in {col}: {value}")
                            break
            
            # @ 기호가 모두 제거되었는지 확인
            self.assertFalse(at_sign_found, "@ signs should be removed from all URLs")
            
            logger.info("@ symbol removal test passed")
        except Exception as e:
            self.fail(f"@ symbol removal test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main() 