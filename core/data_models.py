from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Product:
    """Represents a product, including original input data."""

    # Core identifiers
    id: str
    name: str
    source: str

    # Pricing and Links
    price: float
    url: Optional[str] = None
    image_url: Optional[str] = None

    # Scraping status
    status: Optional[str] = None  # e.g., "OK", "Not Found", "Image Not Found"

    # Source-specific data (from input file or scraping)
    brand: Optional[str] = None
    description: Optional[str] = None
    is_promotional_site: Optional[bool] = None

    # Koryo Gift specific fields
    product_code: Optional[str] = None
    category: Optional[str] = None
    min_order_quantity: Optional[int] = None
    total_price_ex_vat: Optional[float] = None
    total_price_incl_vat: Optional[float] = None
    shipping_info: Optional[str] = None
    specifications: Optional[Dict[str, str]] = None
    quantity_prices: Optional[Dict[str, float]] = None

    # Additional product details
    options: Optional[List[Dict[str, Any]]] = (
        None  # List of available options (e.g., colors, sizes)
    )
    reviews: Optional[List[Dict[str, Any]]] = None  # List of product reviews
    image_gallery: Optional[List[str]] = None  # List of additional product images
    stock_status: Optional[str] = None  # Product stock status
    delivery_time: Optional[str] = None  # Estimated delivery time
    customization_options: Optional[List[str]] = None  # Available customization options

    # --- Fields to store original Haeoreum input data ---
    original_input_data: Dict[str, Any] = field(default_factory=dict)
    # Example keys in original_input_data based on input format:
    # '구분', '담당자', '업체명', '업체코드', '중분류카테고리',
    # '기본수량(1)', '판매단가(V포함)', '본사상품링크', '본사 이미지'
    # --- End Original Input Data Fields ---

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "image_url": self.image_url,
            "brand": self.brand,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "status": self.status,
            "is_promotional_site": self.is_promotional_site,
            "product_code": self.product_code,
            "category": self.category,
            "min_order_quantity": self.min_order_quantity,
            "total_price_ex_vat": self.total_price_ex_vat,
            "total_price_incl_vat": self.total_price_incl_vat,
            "shipping_info": self.shipping_info,
            "specifications": self.specifications,
            "quantity_prices": self.quantity_prices,
            "options": self.options,
            "reviews": self.reviews,
            "image_gallery": self.image_gallery,
            "stock_status": self.stock_status,
            "delivery_time": self.delivery_time,
            "customization_options": self.customization_options,
        }


@dataclass
class MatchResult:
    """Product matching result between source and a potential match."""

    source_product: Product
    matched_product: Product
    text_similarity: float
    image_similarity: float
    combined_similarity: float
    price_difference: float
    price_difference_percent: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "source_product_name": self.source_product.name,
            "source_product_price": self.source_product.price,
            "matched_product": self.matched_product.to_dict(),
            "text_similarity": self.text_similarity,
            "image_similarity": self.image_similarity,
            "combined_similarity": self.combined_similarity,
            "price_difference": self.price_difference,
            "price_difference_percent": self.price_difference_percent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProcessingResult:
    """Overall processing result for a single source product."""

    source_product: Product
    koryo_matches: List[MatchResult] = field(default_factory=list)
    naver_matches: List[MatchResult] = field(default_factory=list)
    best_koryo_match: Optional[MatchResult] = None
    best_naver_match: Optional[MatchResult] = None
    error: Optional[str] = None

    def get_all_matches(self) -> List[MatchResult]:
        return self.koryo_matches + self.naver_matches

    def to_dict(self) -> Dict:
        return {
            "source_product": self.source_product.to_dict(),
            "source_original_data": self.source_product.original_input_data,
            "koryo_matches": [m.to_dict() for m in self.koryo_matches],
            "naver_matches": [m.to_dict() for m in self.naver_matches],
            "best_koryo_match": (
                self.best_koryo_match.to_dict() if self.best_koryo_match else None
            ),
            "best_naver_match": (
                self.best_naver_match.to_dict() if self.best_naver_match else None
            ),
            "error": self.error,
        }
