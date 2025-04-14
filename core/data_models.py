from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class ProductStatus(Enum):
    """Enum representing the status of a product during scraping."""
    OK = "OK"
    NOT_FOUND = "Not Found"
    IMAGE_NOT_FOUND = "Image Not Found"
    FETCH_ERROR = "Fetch Error"
    EXTRACT_ERROR = "Extract Error"
    FETCHED = "Fetched"


@dataclass
class Product:
    """Represents a product, including original input data."""

    # Required fields
    id: str
    name: str
    source: str
    price: float
    
    # Optional fields with default values
    url: str = ""
    image_url: str = ""
    status: str = "OK"
    
    # Optional fields that are commonly used
    brand: str = ""
    description: str = ""
    product_code: str = ""
    category: str = ""
    
    # Pricing related fields
    quantity_prices: Dict[str, float] = field(default_factory=dict)
    min_order_quantity: int = 1
    total_price_ex_vat: float = 0.0
    total_price_incl_vat: float = 0.0
    
    # Additional info
    shipping_info: str = ""
    stock_status: str = ""
    delivery_time: str = ""
    
    # Lists and complex types - lazy loaded
    _image_gallery: Optional[List[str]] = field(default=None, repr=False)
    _specifications: Optional[Dict[str, str]] = field(default=None, repr=False)
    _options: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)
    _reviews: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)
    _customization_options: Optional[List[str]] = field(default=None, repr=False)
    
    # Flags
    is_promotional_site: bool = False
    
    # Original data storage - 메모리 최적화를 위한 JSON 직렬화
    _original_input_data_json: Optional[str] = field(default=None, repr=False)
    _original_input_data: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
    # 캐시 처리용 메타데이터
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Properties for lazy loading
    @property
    def image_gallery(self) -> List[str]:
        if self._image_gallery is None:
            self._image_gallery = []
        return self._image_gallery
    
    @image_gallery.setter
    def image_gallery(self, value: List[str]):
        self._image_gallery = value
    
    @property
    def specifications(self) -> Dict[str, str]:
        if self._specifications is None:
            self._specifications = {}
        return self._specifications
    
    @specifications.setter
    def specifications(self, value: Dict[str, str]):
        self._specifications = value
    
    @property
    def options(self) -> List[Dict[str, Any]]:
        if self._options is None:
            self._options = []
        return self._options
    
    @options.setter
    def options(self, value: List[Dict[str, Any]]):
        self._options = value
    
    @property
    def reviews(self) -> List[Dict[str, Any]]:
        if self._reviews is None:
            self._reviews = []
        return self._reviews
    
    @reviews.setter
    def reviews(self, value: List[Dict[str, Any]]):
        self._reviews = value
    
    @property
    def customization_options(self) -> List[str]:
        if self._customization_options is None:
            self._customization_options = []
        return self._customization_options
    
    @customization_options.setter
    def customization_options(self, value: List[str]):
        self._customization_options = value
    
    @property
    def original_input_data(self) -> Dict[str, Any]:
        # JSON 문자열에서 로드
        if self._original_input_data is None:
            if self._original_input_data_json:
                try:
                    self._original_input_data = json.loads(self._original_input_data_json)
                except Exception:
                    self._original_input_data = {}
            else:
                self._original_input_data = {}
        return self._original_input_data
    
    @original_input_data.setter
    def original_input_data(self, value: Dict[str, Any]):
        # 사전을 직접 설정하고 JSON 문자열도 업데이트
        self._original_input_data = value
        try:
            self._original_input_data_json = json.dumps(value)
        except Exception:
            # 직렬화 불가능한 객체가 있는 경우 기본 사전만 유지
            self._original_input_data_json = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding empty optional fields."""
        result = {
            # Required fields always included
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "price": self.price,
            "status": self.status
        }
        
        # Optional fields only if they have values
        if self.url:
            result["url"] = self.url
        if self.image_url:
            result["image_url"] = self.image_url
        if self.brand:
            result["brand"] = self.brand
        if self.description:
            result["description"] = self.description
        if self.product_code:
            result["product_code"] = self.product_code
        if self.category:
            result["category"] = self.category
            
        # Include non-empty collections - lazy loaded properties
        if self._image_gallery:
            result["image_gallery"] = self._image_gallery
        if self._specifications:
            result["specifications"] = self._specifications
        if self._options:
            result["options"] = self._options
        if self._reviews:
            result["reviews"] = self._reviews
        if self._customization_options:
            result["customization_options"] = self._customization_options
        if self.quantity_prices:
            result["quantity_prices"] = self.quantity_prices
            
        # Include non-default values
        if self.min_order_quantity != 1:
            result["min_order_quantity"] = self.min_order_quantity
        if self.total_price_ex_vat:
            result["total_price_ex_vat"] = self.total_price_ex_vat
        if self.total_price_incl_vat:
            result["total_price_incl_vat"] = self.total_price_incl_vat
        if self.shipping_info:
            result["shipping_info"] = self.shipping_info
        if self.stock_status:
            result["stock_status"] = self.stock_status
        if self.delivery_time:
            result["delivery_time"] = self.delivery_time
        if self.is_promotional_site:
            result["is_promotional_site"] = self.is_promotional_site
            
        # Add original input data
        if self._original_input_data:
            result["original_input_data"] = self._original_input_data
            
        # Add metadata
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """Create a Product instance from a dictionary."""
        # 필수 필드 검사
        required_fields = {"id", "name", "source", "price"}
        if not all(field in data for field in required_fields):
            missing = required_fields - set(data.keys())
            raise ValueError(f"Missing required fields: {missing}")
        
        # 원본 데이터에서 복합 필드 추출 (lazy loading용)
        image_gallery = data.pop("image_gallery", None)
        specifications = data.pop("specifications", None)
        options = data.pop("options", None)
        reviews = data.pop("reviews", None)
        customization_options = data.pop("customization_options", None)
        original_input_data = data.pop("original_input_data", None)
        
        # 기본 객체 생성
        product = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        
        # 복합 필드 설정
        if image_gallery:
            product._image_gallery = image_gallery
        if specifications:
            product._specifications = specifications
        if options:
            product._options = options
        if reviews:
            product._reviews = reviews
        if customization_options:
            product._customization_options = customization_options
        if original_input_data:
            product.original_input_data = original_input_data
            
        return product
        
    def __post_init__(self):
        """데이터클래스 초기화 후 추가 처리"""
        # 문자열 필드 유효성 검사
        self.name = str(self.name) if self.name else ""
        self.url = str(self.url) if self.url else ""
        self.image_url = str(self.image_url) if self.image_url else ""
        
        # 숫자 필드 유효성 검사
        try:
            self.price = float(self.price) if self.price else 0.0
        except (TypeError, ValueError):
            self.price = 0.0


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
