from core.matching.multimodal_matcher import MultiModalMatcher
from core.matching.text_matcher import TextMatcher
from core.matching.image_matcher import ImageMatcher
from core.data_models import Product

def test_multimodal_matcher():
    # Initialize matchers
    text_matcher = TextMatcher()
    image_matcher = ImageMatcher()
    multimodal_matcher = MultiModalMatcher(
        text_matcher=text_matcher,
        image_matcher=image_matcher,
        text_weight=0.6,
        image_weight=0.4
    )
    
    # Test with float values
    text_sim = 0.5
    image_sim = 0.8
    combined_sim = multimodal_matcher.calculate_similarity(text_sim, image_sim)
    print(f"Test with float values: {combined_sim}")
    
    # Calculate expected result manually for comparison
    expected_result = (0.6 * text_sim) + (0.4 * image_sim)
    print(f"Expected result: {expected_result}")
    print(f"Match: {abs(combined_sim - expected_result) < 0.001}")
    
    # Create Product objects for testing
    product1 = Product(
        id="1",
        name="스테인리스 텀블러 500ml",
        price=10000,
        image_url="https://example.com/image1.jpg",
        source="Test"
    )
    
    product2 = Product(
        id="2",
        name="고급 스테인리스 텀블러 500ml",
        price=12000,
        image_url="https://example.com/image2.jpg",
        source="Test"
    )
    
    # Test with Product objects
    # Note: This will need to download images, so it might not work in all environments
    print("\nTest with Product objects:")
    try:
        product_sim = multimodal_matcher.calculate_similarity(product1, product2)
        print(f"Combined similarity: {product_sim}")
    except Exception as e:
        print(f"Error with Product objects: {e}")
    
    # Test with invalid types
    print("\nTest with invalid types:")
    invalid_sim = multimodal_matcher.calculate_similarity("not a product", 123)
    print(f"Result with invalid types: {invalid_sim}")

if __name__ == "__main__":
    test_multimodal_matcher() 