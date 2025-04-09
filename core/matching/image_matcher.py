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
import cv2

from ..data_models import Product
from utils.caching import FileCache, cache_result
from efficientnet_pytorch import EfficientNet
from torchvision import transforms, models
from collections import Counter

class ImageMatcher:
    def __init__(self, cache: Optional[FileCache] = None, similarity_threshold: float = 0.8):
        self.logger = logging.getLogger(__name__)
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        
        # 더 최신 모델로 업그레이드 (b0 → b3)
        try:
            self.model = EfficientNet.from_pretrained('efficientnet-b3')
        except Exception as e:
            self.logger.warning(f"Failed to load EfficientNet-b3, falling back to b0: {e}")
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
            
        self.model.eval()
        
        # 백업 모델로 ResNet 사용 (오픈소스)
        self.backup_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backup_model.eval()
        
        # GPU 사용 (가능한 경우)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.backup_model = self.backup_model.to(self.device)
        
        # 이미지 전처리 파이프라인 (224x224 → 300x300)
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 색상 분석용 색상 정의
        self.color_names = {
            'red': ([0, 100, 100], [10, 255, 255]), # 빨강
            'orange': ([10, 100, 100], [25, 255, 255]), # 주황
            'yellow': ([25, 100, 100], [35, 255, 255]), # 노랑
            'green': ([35, 100, 100], [85, 255, 255]), # 초록
            'blue': ([85, 100, 100], [130, 255, 255]), # 파랑
            'purple': ([130, 100, 100], [170, 255, 255]), # 보라
            'pink': ([170, 100, 100], [180, 255, 255]), # 분홍
            'brown': ([0, 100, 20], [20, 255, 100]), # 갈색
            'white': ([0, 0, 200], [180, 30, 255]), # 하양
            'gray': ([0, 0, 70], [180, 30, 200]), # 회색
            'black': ([0, 0, 0], [180, 30, 70]) # 검정
        }
    
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
            
            # 1. 퍼셉추얼 해시 유사도 (20%)
            hash_sim = self._get_hash_similarity(img1, img2)
            
            # 높은 해시 유사도면 빠른 리턴
            if hash_sim > 0.95:
                return 1.0
            
            # 2. 딥 특징 유사도 (40%)
            feature_sim = self._get_feature_similarity(img1, img2)
            
            # 3. 색상 히스토그램 유사도 (20%) - 새로운 기능
            color_sim = self._get_color_similarity(img1, img2)
            
            # 4. 백업 모델 유사도 (ResNet) (20%) - 새로운 기능
            resnet_sim = self._get_resnet_similarity(img1, img2)
            
            # 결합 유사도 계산 (가중치 조정)
            combined_sim = (0.2 * hash_sim + 
                           0.4 * feature_sim + 
                           0.2 * color_sim + 
                           0.2 * resnet_sim)
            
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
         # 여러 해시 알고리즘 사용 (평균 해시, 차이 해시, 웨이블릿 해시)
         hash1_avg = self._get_image_hash(img1, 'average')
         hash2_avg = self._get_image_hash(img2, 'average')
         
         hash1_phash = self._get_image_hash(img1, 'phash')
         hash2_phash = self._get_image_hash(img2, 'phash')
         
         hash1_whash = self._get_image_hash(img1, 'whash')
         hash2_whash = self._get_image_hash(img2, 'whash')

         if (hash1_avg is None or hash2_avg is None or 
             hash1_phash is None or hash2_phash is None or
             hash1_whash is None or hash2_whash is None):
             return 0.0

         # 각 해시 알고리즘별 유사도 계산
         max_diff_avg = len(hash1_avg.hash) * len(hash1_avg.hash)
         diff_avg = hash1_avg - hash2_avg
         sim_avg = 1 - (diff_avg / max_diff_avg)
         
         max_diff_phash = len(hash1_phash.hash) * len(hash1_phash.hash)
         diff_phash = hash1_phash - hash2_phash
         sim_phash = 1 - (diff_phash / max_diff_phash)
         
         max_diff_whash = len(hash1_whash.hash) * len(hash1_whash.hash)
         diff_whash = hash1_whash - hash2_whash
         sim_whash = 1 - (diff_whash / max_diff_whash)
         
         # 결합 해시 유사도 계산 (가중치 다르게 적용)
         combined_sim = 0.2 * sim_avg + 0.5 * sim_phash + 0.3 * sim_whash
         
         return float(combined_sim)
         
    def _get_image_hash(self, img: Image.Image, hash_type: str = 'average') -> Optional[imagehash.ImageHash]:
        """Gets the hash of an image using specified algorithm, using cache if available."""
        try:
            if hash_type == 'average':
                img_hash = imagehash.average_hash(img)
            elif hash_type == 'phash':
                img_hash = imagehash.phash(img)
            elif hash_type == 'whash':
                img_hash = imagehash.whash(img)
            else:
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
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.extract_features(img_tensor)
                features = torch.mean(features, dim=[2, 3]) # Global average pooling
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}", exc_info=True)
            return None
            
    def _get_color_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """두 이미지의 색상 분포 유사도 계산"""
        try:
            # PIL 이미지를 OpenCV 형식으로 변환
            img1_cv = np.array(img1.convert('RGB'))
            img1_cv = img1_cv[:, :, ::-1].copy()  # RGB -> BGR
            
            img2_cv = np.array(img2.convert('RGB'))
            img2_cv = img2_cv[:, :, ::-1].copy()  # RGB -> BGR
            
            # HSV 색 공간으로 변환
            img1_hsv = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2HSV)
            img2_hsv = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2HSV)
            
            # 색상 분포 계산
            color_dist1 = self._get_color_distribution(img1_hsv)
            color_dist2 = self._get_color_distribution(img2_hsv)
            
            # 색상 분포 유사도 계산
            similarity = 0.0
            total_colors = len(self.color_names)
            
            for color in self.color_names:
                # 각 색상의 비율 차이 계산
                diff = abs(color_dist1.get(color, 0.0) - color_dist2.get(color, 0.0))
                similarity += 1.0 - min(diff, 1.0)  # 차이가 클수록 유사도 감소
                
            # 평균 유사도 반환
            return similarity / total_colors
            
        except Exception as e:
            self.logger.error(f"Error calculating color similarity: {e}", exc_info=True)
            return 0.5  # 오류 발생 시 중간값 반환
            
    def _get_color_distribution(self, img_hsv: np.ndarray) -> Dict[str, float]:
        """이미지의 색상 분포 계산"""
        height, width = img_hsv.shape[:2]
        total_pixels = height * width
        
        color_distribution = {}
        
        for color_name, (lower, upper) in self.color_names.items():
            # HSV 색상 범위에 맞는 마스크 생성
            lower = np.array(lower)
            upper = np.array(upper)
            
            mask = cv2.inRange(img_hsv, lower, upper)
            color_pixels = cv2.countNonZero(mask)
            
            # 해당 색상의 비율 계산
            color_distribution[color_name] = color_pixels / total_pixels
            
        return color_distribution
        
    def _get_resnet_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """ResNet을 사용한 특징 유사도 계산 (백업 모델)"""
        try:
            # 이미지 변환
            img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # ResNet 특징 추출
                features1 = self.backup_model.avgpool(self.backup_model.layer4(
                    self.backup_model.layer3(
                        self.backup_model.layer2(
                            self.backup_model.layer1(
                                self.backup_model.maxpool(
                                    self.backup_model.relu(
                                        self.backup_model.bn1(
                                            self.backup_model.conv1(img1_tensor)
                                        )
                                    )
                                )
                            )
                        )
                    )
                ))
                
                features2 = self.backup_model.avgpool(self.backup_model.layer4(
                    self.backup_model.layer3(
                        self.backup_model.layer2(
                            self.backup_model.layer1(
                                self.backup_model.maxpool(
                                    self.backup_model.relu(
                                        self.backup_model.bn1(
                                            self.backup_model.conv1(img2_tensor)
                                        )
                                    )
                                )
                            )
                        )
                    )
                ))
                
                # 특징 평탄화
                features1 = torch.flatten(features1, 1)
                features2 = torch.flatten(features2, 1)
                
                # 코사인 유사도 계산
                similarity = torch.nn.functional.cosine_similarity(features1, features2)
                
            return float(similarity.cpu().numpy())
            
        except Exception as e:
            self.logger.error(f"Error calculating ResNet similarity: {e}", exc_info=True)
            return 0.5  # 오류 발생 시 중간값 반환

    def _download_and_preprocess(self, url: str) -> Optional[Image.Image]:
        """Download and preprocess image from URL"""
        if not url or not url.startswith('http'):
            self.logger.warning(f"Invalid image URL: {url}")
            return None
            
        try:
            # Download image (타임아웃 증가 및 리트라이 추가)
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
            except requests.exceptions.RequestException:
                # 실패시 1회 재시도
                self.logger.warning(f"Retrying download for URL: {url}")
                response = requests.get(url, timeout=20)
                response.raise_for_status()
            
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