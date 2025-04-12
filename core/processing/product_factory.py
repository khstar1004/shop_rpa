import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field

import pandas as pd

from ..data_models import Product
from .data_cleaner import DataCleaner


@dataclass
class Product:
    """제품 정보를 담는 데이터 클래스"""
    id: str
    name: str
    price: float
    source: str  # 'koryo', 'other_source' 등
    url: str
    image_url: str = ""
    image_gallery: List[str] = field(default_factory=list)
    product_code: str = ""
    description: str = ""
    specifications: Dict[str, str] = field(default_factory=dict)
    quantity_prices: Dict[str, float] = field(default_factory=dict)
    original_input_data: Dict = field(default_factory=dict)
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, success, failed
    error_message: str = ""
    koryo_name: str = ""
    koryo_price: float = 0
    koryo_image_url: str = ""
    koryo_url: str = ""

    def __post_init__(self):
        """초기화 후 데이터 검증 및 기본값 설정"""
        # 필수 필드 검증
        if not self.id or not self.name or not self.source:
            raise ValueError("Required fields (id, name, source) must not be empty")
            
        # 가격이 없으면 0으로 설정
        if self.price is None:
            self.price = 0
            
        # URL이 없으면 빈 문자열로 설정
        if not self.url:
            self.url = ""
            
        # 이미지 URL이 없고 갤러리가 있으면 첫 번째 이미지를 메인 이미지로 설정
        if not self.image_url and self.image_gallery:
            self.image_url = self.image_gallery[0]
            
        # 수집 시간이 없으면 현재 시간으로 설정
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """제품 정보를 딕셔너리로 변환"""
        return {
            'id': self.id,
            'name': self.name,
            'price': self.price,
            'source': self.source,
            'url': self.url,
            'image_url': self.image_url,
            'image_gallery': self.image_gallery,
            'product_code': self.product_code,
            'description': self.description,
            'specifications': self.specifications,
            'quantity_prices': self.quantity_prices,
            'fetched_at': self.fetched_at,
            'status': self.status,
            'error_message': self.error_message,
            'koryo_name': self.koryo_name,
            'koryo_price': self.koryo_price,
            'koryo_image_url': self.koryo_image_url,
            'koryo_url': self.koryo_url
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Product':
        """딕셔너리에서 제품 객체 생성"""
        required_fields = {'id', 'name', 'source'}
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(data.keys())}")
            
        return cls(
            id=data['id'],
            name=data['name'],
            price=float(data.get('price', 0)),
            source=data['source'],
            url=data.get('url', ''),
            image_url=data.get('image_url', ''),
            image_gallery=data.get('image_gallery', []),
            product_code=data.get('product_code', ''),
            description=data.get('description', ''),
            specifications=data.get('specifications', {}),
            quantity_prices=data.get('quantity_prices', {}),
            fetched_at=data.get('fetched_at', datetime.now().isoformat()),
            status=data.get('status', 'pending'),
            error_message=data.get('error_message', ''),
            koryo_name=data.get('koryo_name', ''),
            koryo_price=float(data.get('koryo_price', 0)),
            koryo_image_url=data.get('koryo_image_url', ''),
            koryo_url=data.get('koryo_url', '')
        )

    def validate(self) -> bool:
        """제품 데이터 유효성 검사"""
        try:
            # 필수 필드 검사
            if not self.id or not self.name or not self.source:
                self.error_message = "필수 필드 누락"
                self.status = "failed"
                return False
                
            # 가격 검사
            if self.price < 0:
                self.error_message = "잘못된 가격"
                self.status = "failed"
                return False
                
            # 이미지 URL 검사
            if not self.image_url and not self.image_gallery:
                self.error_message = "이미지 없음"
                self.status = "failed"
                return False
                
            # 모든 검사 통과
            self.status = "success"
            return True
            
        except Exception as e:
            self.error_message = f"유효성 검사 중 오류: {str(e)}"
            self.status = "failed"
            return False

    def get_status_message(self) -> str:
        """현재 상태 메시지 반환"""
        if self.status == "success":
            return "성공"
        elif self.status == "failed":
            return f"실패: {self.error_message}"
        else:
            return "처리 중"


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

            # 기본 Product 객체 생성
            product = Product(
                id=product_code,
                name=product_name,
                price=price,
                source="haeoreum",  # 기본 출처
                url="",  # 기본값으로 빈 문자열 설정
                original_input_data=row.to_dict(),  # 원본 데이터 보존
            )

            # 선택적 필드 설정
            if (
                "본사 이미지" in row
                and not pd.isna(row["본사 이미지"])
                and str(row["본사 이미지"]).strip()
            ):
                image_url = self.data_cleaner.clean_url(str(row["본사 이미지"]).strip(), True)
                product.image_url = image_url
                product.original_input_data["본사 이미지"] = image_url

            if (
                "본사상품링크" in row
                and not pd.isna(row["본사상품링크"])
                and str(row["본사상품링크"]).strip()
            ):
                product.url = self.data_cleaner.clean_url(
                    str(row["본사상품링크"]).strip(), True
                )

            # 추가 정보가 있으면 설정
            if "공급사명" in row and not pd.isna(row["공급사명"]):
                product.brand = str(row["공급사명"]).strip()

            if "중분류카테고리" in row and not pd.isna(row["중분류카테고리"]):
                product.category = str(row["중분류카테고리"]).strip()

            # 고려기프트 관련 필드 설정
            if "고려기프트 상품명" in row and not pd.isna(row["고려기프트 상품명"]):
                product.koryo_name = str(row["고려기프트 상품명"]).strip()
            else:
                product.koryo_name = "유사상품 없음"

            if "고려기프트 가격" in row and not pd.isna(row["고려기프트 가격"]):
                try:
                    price_str = str(row["고려기프트 가격"]).strip().replace(",", "").replace("원", "")
                    product.koryo_price = float(price_str)
                except ValueError:
                    self.logger.warning(f"Invalid Koryo price format: {row['고려기프트 가격']}")
                    product.koryo_price = 0
            else:
                product.koryo_price = 0

            if "고려기프트 이미지" in row and not pd.isna(row["고려기프트 이미지"]):
                product.koryo_image_url = self.data_cleaner.clean_url(str(row["고려기프트 이미지"]).strip(), True)
            else:
                product.koryo_image_url = "유사상품 없음"

            if "고려기프트 상품링크" in row and not pd.isna(row["고려기프트 상품링크"]):
                product.koryo_url = self.data_cleaner.clean_url(str(row["고려기프트 상품링크"]).strip(), True)
            else:
                product.koryo_url = "유사상품 없음"

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
                    original_input_data={"error": str(e)},
                )
            except:
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
