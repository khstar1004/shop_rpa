import logging
import os
import requests
from io import BytesIO
from typing import Optional, Dict, List, Any, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import imagehash
from rembg import remove

from ..data_models import Product
from utils.caching import FileCache, cache_result
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

class ImageMatcher:
    def __init__(self, cache: Optional[FileCache] = None, similarity_threshold: float = 0.8):
        self.logger = logging.getLogger(__name__)
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        
        # Initialize EfficientNet model
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def calculate_similarity(self, url1: Optional[str], url2: Optional[str]) -> float:
        if not url1 or not url2:
            return 0.0
            
        # Ensure consistent key order
        if url1 > url2:
             url1, url2 = url2, url1
             
        if self.cache:
            return self._cached_calculate_similarity(url1, url2)
        else:
            return self._calculate_similarity_logic(url1, url2)

    def _cached_calculate_similarity(self, url1: str, url2: str) -> float:
        @cache_result(self.cache, key_prefix="image_sim")
        def cached_logic(u1, u2):
            return self._calculate_similarity_logic(u1, u2)
        return cached_logic(url1, url2)

    def _calculate_similarity_logic(self, url1: str, url2: str) -> float:
        """Core logic for calculating image similarity"""
        try:
            # Download and preprocess images (cached)
            img1 = self._get_processed_image(url1)
            img2 = self._get_processed_image(url2)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Calculate perceptual hash similarity (cached)
            hash_sim = self._get_hash_similarity(img1, img2)
            
            if hash_sim > 0.95:
                return 1.0
            
            # Calculate deep feature similarity (cached)
            feature_sim = self._get_feature_similarity(img1, img2)
            
            combined_sim = 0.3 * hash_sim + 0.7 * feature_sim
            return float(combined_sim)
            
        except Exception as e:
            self.logger.error(f"Error calculating image similarity between {url1} and {url2}: {str(e)}", exc_info=True)
            return 0.0

    def _get_processed_image(self, url: str) -> Optional[Image.Image]:
        """Downloads, preprocesses (removes bg), and caches the image."""
        if self.cache:
            cache_key = f"processed_image|{url}"
            cached_image = self.cache.get(cache_key)
            if cached_image is not None:
                return cached_image
        
        img = self._download_and_preprocess(url)
        if img and self.cache:
            self.cache.set(cache_key, img)
        return img
        
    def _get_hash_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
         """Calculates hash similarity, potentially using cached hashes."""
         hash1 = self._get_image_hash(img1)
         hash2 = self._get_image_hash(img2)

         if hash1 is None or hash2 is None:
             return 0.0

         max_diff = len(hash1.hash) * len(hash1.hash)
         diff = hash1 - hash2
         similarity = 1 - (diff / max_diff)
         return float(similarity)
         
    def _get_image_hash(self, img: Image.Image) -> Optional[imagehash.ImageHash]:
        """Gets the average hash of an image, using cache if available."""
        # Need a way to uniquely identify the image object for caching its hash
        # Hashing the image bytes itself can be slow. Using URL if available, else skip hash caching
        # This assumes _get_processed_image was called with a URL.
        # For more robust caching, pass URL or a unique ID here.
        try:
             img_hash = imagehash.average_hash(img)
             return img_hash
        except Exception as e:
             self.logger.warning(f"Could not calculate image hash: {e}")
             return None

    def _get_feature_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates feature similarity, potentially using cached features."""
        features1 = self._get_image_features(img1)
        features2 = self._get_image_features(img2)
        
        if features1 is None or features2 is None:
             return 0.0

        similarity = torch.nn.functional.cosine_similarity(features1, features2)
        return float(similarity.cpu().numpy())

    def _get_image_features(self, img: Image.Image) -> Optional[torch.Tensor]:
        """Extracts deep features from an image, using cache if available."""
        # Similar caching challenge as with _get_image_hash. 
        # Requires a stable identifier (like URL) passed down or calculated.
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.extract_features(img_tensor)
                features = torch.mean(features, dim=[2, 3]) # Global average pooling
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}", exc_info=True)
            return None

    def _download_and_preprocess(self, url: str) -> Optional[Image.Image]:
        """Download and preprocess image from URL"""
        if not url or not url.startswith('http'):
            self.logger.warning(f"Invalid image URL: {url}")
            return None
            
        try:
            # Download image
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises exception for 4XX/5XX responses
            
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Basic image validation
            if img.width < 10 or img.height < 10:
                self.logger.warning(f"Image too small: {img.width}x{img.height} for URL: {url}")
                return img  # Return as is, don't try background removal
            
            try:
                # Try to remove background, but continue if it fails
                img_no_bg = remove(img)
                
                # Convert back to PIL Image if needed
                if isinstance(img_no_bg, bytes):
                    img_no_bg = Image.open(BytesIO(img_no_bg))
                
                return img_no_bg
            except Exception as bg_error:
                self.logger.warning(f"Background removal failed for {url}: {str(bg_error)}. Using original image.")
                return img  # Return original image if background removal fails
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error downloading image from {url}: {str(e)}")
            return None
        except (IOError, OSError) as e:
            self.logger.error(f"Error processing image from {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error handling image from {url}: {str(e)}", exc_info=True)
            return None 