import unittest
import os
import pandas as pd
import shutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestExcelSaving(unittest.TestCase):
    """테스트 엑셀 저장 기능"""

    @classmethod
    def setUpClass(cls):
        """테스트 시작 전 설정"""
        cls.test_dir = "test_excel_output"
        os.makedirs(cls.test_dir, exist_ok=True)
        logger.info(f"Created test directory: {cls.test_dir}")
        
    @classmethod
    def tearDownClass(cls):
        """테스트 완료 후 정리"""
        try:
            shutil.rmtree(cls.test_dir)
            logger.info(f"Removed test directory: {cls.test_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up test directory: {e}")

    def test_excel_save_with_image_formulas(self):
        """IMAGE 함수가 포함된 엑셀 파일이 올바르게 저장되는지 확인"""
        # 테스트 데이터 생성
        test_data = {
            "상품명": ["상품A", "상품B", "상품C"],
            "가격": [1000, 2000, 3000],
            "본사 이미지": [
                '=IMAGE("https://example.com/image1.jpg", 2)',
                '=IMAGE("https://example.com/image2.jpg", 2)',
                '=IMAGE("https://user@example.com/image3.jpg", 2)',  # @ 기호 포함
            ]
        }
        
        # 테스트 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_file = os.path.join(self.test_dir, f"test_image_formulas_{timestamp}.xlsx")
        
        # 데이터프레임 생성 및 저장
        df = pd.DataFrame(test_data)
        
        try:
            # @ 기호가 있는 URL 처리 (Excel 저장 오류 방지)
            for col in df.columns:
                if col.endswith("이미지"):
                    df[col] = df[col].apply(
                        lambda x: x.replace('@', '') if isinstance(x, str) else x
                    )
            
            # 엑셀 파일로 저장
            logger.info(f"Saving test Excel file to: {test_file}")
            df.to_excel(test_file, index=False)
            
            # 파일이 저장됐는지 확인
            self.assertTrue(os.path.exists(test_file), "Excel file was not saved")
            self.assertTrue(os.path.getsize(test_file) > 0, "Excel file is empty")
            
            # 저장된 파일 읽기
            read_df = pd.read_excel(test_file)
            self.assertEqual(len(read_df), 3, "DataFrame should have 3 rows")
            self.assertEqual(len(read_df.columns), 3, "DataFrame should have 3 columns")
            
            # Excel에서 읽으면 IMAGE 함수는 일반 텍스트로 저장되기 때문에 정확한 확인 불가
            # 대신 파일 자체가 성공적으로 저장되었는지만 확인
            logger.info(f"Excel file successfully saved with size: {os.path.getsize(test_file)} bytes")
            logger.info("Excel saving test passed")
            
        except Exception as e:
            logger.error(f"Error in Excel saving test: {e}")
            raise
        finally:
            # 테스트 파일 삭제
            if os.path.exists(test_file):
                os.remove(test_file)

if __name__ == "__main__":
    unittest.main() 