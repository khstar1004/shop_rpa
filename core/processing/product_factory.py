import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional, Union, List

import pandas as pd

from ..data_models import Product, ProductStatus
from .data_cleaner import DataCleaner


class ProductFactory:
    """Product 객체 생성을 담당하는 클래스"""

    def __init__(
        self,
        config: Dict,
        logger: Optional[logging.Logger] = None,
        data_cleaner: Optional[DataCleaner] = None,
    ):
        """
        상품 팩토리 초기화

        Args:
            config: 애플리케이션 설정
            logger: 로깅 인스턴스
            data_cleaner: 데이터 정제기 인스턴스 (없으면 생성)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.data_cleaner = data_cleaner or DataCleaner(config, self.logger)

    def create_product_from_row(self, row: pd.Series) -> Optional[Product]:
        """
        DataFrame 행에서 Product 객체 생성

        Args:
            row: DataFrame의 한 행

        Returns:
            Product 객체 또는 None (오류 시)
        """
        try:
            # 필수 필드 확인
            required_fields = ["상품Code", "상품명", "판매단가(V포함)"]
            for field in required_fields:
                if field not in row or pd.isna(row[field]):
                    self.logger.warning(
                        f"Required field '{field}' missing in product data"
                    )

            # 제품 코드 확인
            product_code = None
            if "상품Code" in row and not pd.isna(row["상품Code"]):
                product_code = str(row["상품Code"]).strip()
            elif "Code" in row and not pd.isna(row["Code"]):
                product_code = str(row["Code"]).strip()
            elif "상품코드" in row and not pd.isna(row["상품코드"]):
                product_code = str(row["상품코드"]).strip()

            if not product_code:
                # 고유 코드 생성
                product_code = f"GEN-{int(datetime.now().timestamp())}"
                self.logger.warning(f"No product code found, generated: {product_code}")

            # 제품명 확인
            product_name = None
            if "상품명" in row and not pd.isna(row["상품명"]):
                product_name = self.data_cleaner.clean_product_names(str(row["상품명"]))
            else:
                product_name = "유사상품 없음"  # 상품명이 없을 때 "유사상품 없음"으로 설정

            if not product_name:
                self.logger.error("No product name found, cannot create product")
                return None

            # 가격 확인
            price = 0
            if "판매단가(V포함)" in row and not pd.isna(row["판매단가(V포함)"]):
                try:
                    price_str = (
                        str(row["판매단가(V포함)"])
                        .strip()
                        .replace(",", "")
                        .replace("원", "")
                    )
                    price = float(price_str)
                except ValueError:
                    self.logger.warning(
                        f"Invalid price format: {row['판매단가(V포함)']}, using 0"
                    )
                    price = 0

            # 이미지 URL
            image_url = ""
            if (
                "본사 이미지" in row
                and not pd.isna(row["본사 이미지"])
                and str(row["본사 이미지"]).strip()
            ):
                image_url = self.data_cleaner.clean_url(str(row["본사 이미지"]).strip(), True)

            # 제품 URL
            product_url = ""
            if (
                "본사상품링크" in row
                and not pd.isna(row["본사상품링크"])
                and str(row["본사상품링크"]).strip()
            ):
                product_url = self.data_cleaner.clean_url(
                    str(row["본사상품링크"]).strip(), True
                )

            # 상태 확인 및 설정
            product_status = None
            if not product_name:
                product_status = ProductStatus.NOT_FOUND
            elif not price:
                product_status = "Price Not Found"  # Price not found
            elif not image_url:
                product_status = ProductStatus.IMAGE_NOT_FOUND
            else:
                product_status = ProductStatus.OK

            # 기본 Product 객체 생성
            product = Product(
                id=product_code,
                name=product_name,
                price=price,
                source="haeoreum",  # 기본 출처
                url=product_url,  # 제품 URL
                image_url=image_url,  # 이미지 URL
                original_input_data=row.to_dict(),  # 원본 데이터 보존
                status=product_status  # 상태 설정
            )

            # 추가 정보가 있으면 설정
            if "공급사명" in row and not pd.isna(row["공급사명"]):
                product.brand = str(row["공급사명"]).strip()

            if "중분류카테고리" in row and not pd.isna(row["중분류카테고리"]):
                product.category = str(row["중분류카테고리"]).strip()

            # 고려기프트 관련 필드 설정 (기존 자체 Product 클래스의 field 제거)
            # product.koryo_name = ...
            # product.koryo_price = ...
            # product.koryo_image_url = ...
            # product.koryo_url = ...

            return product

        except Exception as e:
            self.logger.error(
                f"Error creating Product from row: {str(e)}", exc_info=True
            )

            # 실패 시 최소한의 Product 생성 시도
            try:
                fallback_id = f"ERROR-{int(datetime.now().timestamp())}"

                # 행 인덱스 또는 첫 번째 컬럼을 이름으로 사용
                fallback_name = "Unknown Product"
                if isinstance(row.index, pd.RangeIndex):
                    fallback_name = f"Product #{row.name}"
                elif len(row) > 0:
                    fallback_name = f"Product from row {row.name}"

                return Product(
                    id=fallback_id,
                    name=fallback_name,
                    price=0.0,
                    source="haeoreum",
                    url="",
                    status=ProductStatus.EXTRACT_ERROR,  # 오류 상태 설정
                    original_input_data={"error": str(e)},
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback product creation failed: {fallback_error}")
                return None

    def create_products_from_dataframe(self, df: pd.DataFrame) -> list:
        """
        데이터프레임에서 Product 객체 리스트 생성

        Args:
            df: 제품 정보가 있는 DataFrame

        Returns:
            Product 객체 리스트
        """
        products = []

        # 각 행에 대해 Product 생성
        for _, row in df.iterrows():
            product = self.create_product_from_row(row)
            if product:
                products.append(product)

        self.logger.info(
            f"Created {len(products)} products from DataFrame with {len(df)} rows"
        )
        return products
