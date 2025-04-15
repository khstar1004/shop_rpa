import concurrent.futures
import hashlib
import io
import logging
import os
import threading
from collections import Counter
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
import requests
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image, UnidentifiedImageError
from rembg import remove
from torchvision import models, transforms

from utils.caching import FileCache, cache_result

from ..data_models import Product, MatchResult
from core.matching.base_matcher import BaseMatcher


class ImageMatcher(BaseMatcher):
    """Image-based product matching implementation"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.matching_settings = config.get("IMAGE_MATCHER", {})
        self.image_similarity_threshold = float(self.matching_settings.get("image_similarity_threshold", 0.70))
        self.image_weight = float(self.matching_settings.get("image_weight", 0.3))
        
        self.cache = config.get("cache", None)
        self.remove_background = bool(self.matching_settings.get("remove_background", False))
        self.max_image_dimension = self.matching_settings.get("max_image_dimension", None)
        if self.max_image_dimension:
            self.max_image_dimension = int(self.max_image_dimension)

        # Initialize models to None for lazy loading
        self._model = None
        self._backup_model = None
        self._model_lock = threading.Lock()
        self._backup_model_lock = threading.Lock()

        # SIFT 및 AKAZE 특징 추출기 초기화
        self.sift = cv2.SIFT_create()
        self.akaze = cv2.AKAZE_create()

        # FLANN 매처 설정
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # BFMatcher 초기화
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf_knn = cv2.BFMatcher()

        # GPU 사용 (가능한 경우)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 이미지 전처리 파이프라인 (224x224 → 300x300)
        self.transform = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 색상 분석용 색상 정의
        self.color_names = {
            "red": ([0, 100, 100], [10, 255, 255]),  # 빨강
            "orange": ([10, 100, 100], [25, 255, 255]),  # 주황
            "yellow": ([25, 100, 100], [35, 255, 255]),  # 노랑
            "green": ([35, 100, 100], [85, 255, 255]),  # 초록
            "blue": ([85, 100, 100], [130, 255, 255]),  # 파랑
            "purple": ([130, 100, 100], [170, 255, 255]),  # 보라
            "pink": ([170, 100, 100], [180, 255, 255]),  # 분홍
            "brown": ([0, 100, 20], [20, 255, 100]),  # 갈색
            "white": ([0, 0, 200], [180, 30, 255]),  # 하양
            "gray": ([0, 0, 70], [180, 30, 200]),  # 회색
            "black": ([0, 0, 0], [180, 30, 70]),  # 검정
        }

    @property
    def model(self):
        """Lazy load the EfficientNet model."""
        if self._model is None:
            with self._model_lock:
                if self._model is None: # Double-check locking
                    try:
                        self.logger.info("Lazy loading EfficientNet-b3 model...")
                        model_instance = EfficientNet.from_pretrained("efficientnet-b3")
                        model_instance.eval()
                        self._model = model_instance.to(self.device)
                        self.logger.info("EfficientNet-b3 model loaded successfully.")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load EfficientNet-b3, falling back to b0: {e}"
                        )
                        try:
                            model_instance = EfficientNet.from_pretrained("efficientnet-b0")
                            model_instance.eval()
                            self._model = model_instance.to(self.device)
                            self.logger.info("EfficientNet-b0 model loaded successfully.")
                        except Exception as fallback_e:
                             self.logger.error(f"Failed to load even EfficientNet-b0: {fallback_e}", exc_info=True)
                             # Set to a dummy value or raise an error depending on desired behavior
                             self._model = None # Or raise RuntimeError("Could not load any EfficientNet model")
        return self._model

    @property
    def backup_model(self):
        """Lazy load the ResNet backup model."""
        if self._backup_model is None:
            with self._backup_model_lock:
                 if self._backup_model is None: # Double-check locking
                    try:
                        self.logger.info("Lazy loading ResNet50 backup model...")
                        model_instance = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                        model_instance.eval()
                        self._backup_model = model_instance.to(self.device)
                        self.logger.info("ResNet50 backup model loaded successfully.")
                    except Exception as e:
                        self.logger.error(f"Failed to load ResNet50 backup model: {e}", exc_info=True)
                        self._backup_model = None # Or raise
        return self._backup_model

    def match(self, source_image: str, target_image: str) -> float:
        """Calculate image similarity score between source and target images"""
        try:
            # Preprocess images
            source_features = self._extract_features(source_image)
            target_features = self._extract_features(target_image)
            
            # Calculate similarity score
            similarity = self._calculate_similarity(source_features, target_features)
            
            # Apply threshold and weight
            if similarity >= self.image_similarity_threshold:
                return similarity * self.image_weight
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error in image matching: {str(e)}")
            return 0.0

    def calculate_similarity(
        self, url1: Optional[str], url2: Optional[str], max_dimension: int = None
    ) -> float:
        """
        두 이미지의 유사도를 계산합니다.

        Args:
            url1: 첫 번째 이미지 URL
            url2: 두 번째 이미지 URL
            max_dimension: 이미지 처리를 위한 최대 해상도 (속도 최적화)

        Returns:
            유사도 점수 (0.0 ~ 1.0)
        """
        if not url1 or not url2:
            self.logger.debug(f"Cannot calculate similarity, one or both URLs are missing: {url1}, {url2}")
            return 0.0

        # Ensure consistent key order for caching
        if url1 > url2:
            url1, url2 = url2, url1

        # 캐시 키 생성 (해시 사용)
        cache_key_string = f"{url1}|{url2}|{max_dimension}"
        cache_key = hashlib.md5(cache_key_string.encode()).hexdigest()

        if self.cache:
            # Pass the generated key directly
            return self._cached_calculate_similarity(url1, url2, max_dimension, cache_key)
        else:
            # Calculate directly if no cache
            return self._calculate_similarity_logic(url1, url2, max_dimension)

    def _cached_calculate_similarity(
        self,
        url1: str,
        url2: str,
        max_dimension: int,
        cache_key: str, # Accept pre-generated key
    ) -> float:
        """캐시를 활용한 이미지 유사도 계산"""

        # Define the logic function separately
        def logic_to_cache(u1, u2, max_dim):
            return self._calculate_similarity_logic(u1, u2, max_dim)

        # Apply the cache decorator using the provided key
        cached_logic = cache_result(self.cache, key=cache_key)(logic_to_cache)

        return cached_logic(url1, url2, max_dimension)

    def _calculate_similarity_logic(
        self, url1: str, url2: str, max_dimension: int = None
    ) -> float:
        """Core logic for calculating image similarity"""
        try:
            # 병렬로 이미지 다운로드 및 전처리
            img1, img2 = None, None
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(
                    self._get_processed_image, url1, max_dimension
                )
                future2 = executor.submit(
                    self._get_processed_image, url2, max_dimension
                )
                img1 = future1.result()
                img2 = future2.result()

            if img1 is None or img2 is None:
                self.logger.warning(f"Could not process one or both images for similarity calculation: {url1}, {url2}")
                return 0.0

            # 높은 성능: 이미지가 너무 유사한 경우 (동일한 이미지) - 빠른 체크
            if self._check_exact_match(img1, img2):
                 self.logger.debug(f"Exact match detected for {url1} and {url2}")
                 return 1.0

            # 1. 퍼셉추얼 해시 유사도 (15%)
            hash_sim = self._get_hash_similarity(img1, img2)

            # 높은 해시 유사도면 빠른 리턴 - 성능 최적화
            if hash_sim > 0.95:
                adjusted_sim = 0.95 + (hash_sim - 0.95) * 0.5 # Map 0.95-1.0 to 0.95-0.975 for slight diff
                self.logger.debug(f"High hash similarity ({hash_sim:.3f}), returning early: {adjusted_sim:.3f}")
                return adjusted_sim

            # 2. 딥 특징 유사도 (25%)
            feature_sim = self._get_feature_similarity(img1, img2)

            # 3. 색상 히스토그램 유사도 (10%)
            color_sim = self._get_color_similarity(img1, img2)

            # 4. SIFT + RANSAC 유사도 (30%)
            sift_ransac_sim = self._get_sift_ransac_similarity(img1, img2)

            # 5. AKAZE 유사도 (10%)
            akaze_sim = self._get_akaze_similarity(img1, img2)

            # 6. 백업 모델 유사도 (ResNet) (10%)
            # 성능 최적화: 앞의 유사도가 낮으면 생략
            current_avg_sim = (hash_sim + feature_sim + color_sim + sift_ransac_sim + akaze_sim) / 5
            if current_avg_sim < 0.3: # Threshold to skip heavy backup model
                resnet_sim = 0.0
                self.logger.debug(f"Skipping ResNet similarity calculation due to low current avg sim ({current_avg_sim:.3f}) for {url1}, {url2}")
            else:
                resnet_sim = self._get_resnet_similarity(img1, img2)

            # 결합 유사도 계산 (가중치 확인 및 조정 필요)
            weights = {
                'hash': 0.15, 'feature': 0.25, 'color': 0.10,
                'sift': 0.30, 'akaze': 0.10, 'resnet': 0.10
            }
            # Ensure weights sum to 1
            assert abs(sum(weights.values()) - 1.0) < 1e-6, "Similarity weights do not sum to 1"

            combined_sim = (
                weights['hash'] * hash_sim
                + weights['feature'] * feature_sim
                + weights['color'] * color_sim
                + weights['sift'] * sift_ransac_sim
                + weights['akaze'] * akaze_sim
                + weights['resnet'] * resnet_sim
            )

            self.logger.debug(f"Similarity details for ({url1}, {url2}): "
                             f"Hash={hash_sim:.3f}, Feature={feature_sim:.3f}, Color={color_sim:.3f}, "
                             f"SIFT={sift_ransac_sim:.3f}, AKAZE={akaze_sim:.3f}, ResNet={resnet_sim:.3f}, "
                             f"Combined={combined_sim:.3f}")

            return float(combined_sim)

        except Exception as e:
            self.logger.error(
                f"Error calculating image similarity logic between {url1} and {url2}: {str(e)}",
                exc_info=True,
            )
            return 0.0

    def _get_sift_ransac_similarity(
        self, img1: Image.Image, img2: Image.Image
    ) -> float:
        """SIFT 특징과 RANSAC을 사용한 유사도 계산"""
        try:
            # PIL 이미지를 OpenCV 형식으로 변환
            img1_cv = np.array(img1.convert("RGB"))
            img1_cv = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2GRAY)

            img2_cv = np.array(img2.convert("RGB"))
            img2_cv = cv2.cvtColor(img2_cv, cv2.COLOR_RGB2GRAY)

            # DoG(Difference of Gaussian) 계산
            ksize1, ksize2 = 3, 11
            img1_dog = self._calculate_dog(img1_cv, ksize1, ksize2)
            img2_dog = self._calculate_dog(img2_cv, ksize1, ksize2)

            # SIFT 키포인트 및 디스크립터 추출
            kp1, des1 = self.sift.detectAndCompute(img1_dog, None)
            kp2, des2 = self.sift.detectAndCompute(img2_dog, None)

            # 키포인트나 디스크립터가 없으면 0 반환
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return 0.0

            # KNN 매칭
            matches = self.bf_knn.knnMatch(des1, des2, k=2)

            # Lowe's ratio 테스트를 통한 좋은 매치 선별
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 충분한 매치가 있는지 확인
            if len(good_matches) < 5:
                return 0.0

            # RANSAC을 사용한 호모그래피 계산을 위한 포인트 추출
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            # RANSAC으로 호모그래피 계산
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # 인라이어 매치 (RANSAC 테스트를 통과한 매치)
            inlier_matches = [
                m for i, m in enumerate(good_matches) if matches_mask[i] == 1
            ]

            # 매칭 점수 계산
            # 인라이어 매치가 없으면 0, 아니면 인라이어 매치 비율
            if len(inlier_matches) == 0 or len(good_matches) == 0:
                return 0.0

            matching_score = len(inlier_matches) / len(good_matches)

            # 점수 정규화 (0.3 이상이면 의미있는 매치로 간주)
            if matching_score >= 0.3:
                # 0.3~1.0 범위를 0.5~1.0 범위로 확장
                normalized_score = 0.5 + (matching_score - 0.3) * 0.5 / 0.7
                return min(1.0, max(0.0, normalized_score))
            else:
                # 낮은 점수는 그대로 유지하되 가중치 낮춤
                return matching_score * 0.5

        except Exception as e:
            self.logger.error(
                f"Error calculating SIFT+RANSAC similarity: {e}", exc_info=True
            )
            return 0.0

    def _calculate_dog(self, image, ksize1, ksize2):
        """DoG (Difference of Gaussian) 계산"""
        blurred1 = cv2.GaussianBlur(image, (ksize1, ksize1), 0)
        blurred2 = cv2.GaussianBlur(image, (ksize2, ksize2), 0)
        dog = blurred1 - blurred2
        return dog

    def _get_akaze_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """AKAZE 특징을 사용한 유사도 계산"""
        try:
            # PIL 이미지를 OpenCV 형식으로 변환
            img1_cv = np.array(img1.convert("RGB"))
            img1_cv = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2GRAY)

            img2_cv = np.array(img2.convert("RGB"))
            img2_cv = cv2.cvtColor(img2_cv, cv2.COLOR_RGB2GRAY)

            # AKAZE 키포인트 및 디스크립터 추출
            kp1, des1 = self.akaze.detectAndCompute(img1_cv, None)
            kp2, des2 = self.akaze.detectAndCompute(img2_cv, None)

            # 키포인트나 디스크립터가 없으면 0 반환
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return 0.0

            # 브루트포스 매칭
            matches = self.bf.match(des1, des2)

            # 거리순으로 정렬
            matches = sorted(matches, key=lambda x: x.distance)

            # 매치가 없으면 0 반환
            if len(matches) == 0:
                return 0.0

            # 매칭 점수 계산
            # 최소 5개 이상의 매치가 있거나 첫 번째 매치의 거리가 충분히 작으면 의미있는 매치로 간주
            match_param = 50

            if len(matches) > 4 or (
                len(matches) > 0 and matches[0].distance < match_param
            ):
                # 거리가 작을수록 유사도가 높음
                min_distance = min(
                    match_param,
                    matches[0].distance if len(matches) > 0 else match_param,
                )
                similarity = 1.0 - (min_distance / match_param)

                # 유사도 점수 스케일링
                if len(matches) > 4:
                    # 더 많은 매치가 있으면 보너스 점수
                    similarity = min(1.0, similarity * 1.2)

                return similarity
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating AKAZE similarity: {e}", exc_info=True)
            return 0.0

    def _check_exact_match(self, img1: Image.Image, img2: Image.Image) -> bool:
        """빠른 동일 이미지 체크 (비용이 적은 방법)"""
        # 크기가 다르면 동일하지 않음
        if img1.size != img2.size:
            return False

        # 해상도가 낮으면 픽셀 직접 비교
        if img1.width * img1.height < 10000:  # 100x100 픽셀 이하
            try:
                # NumPy 배열로 변환하여 비교
                arr1 = np.array(img1)
                arr2 = np.array(img2)
                return np.array_equal(arr1, arr2)
            except:
                return False

        # 더 큰 이미지는 이미지 해시만 비교 (더 빠름)
        try:
            hash1 = imagehash.average_hash(img1, hash_size=8)
            hash2 = imagehash.average_hash(img2, hash_size=8)
            return hash1 - hash2 < 5  # 거의 동일한 이미지 (약간의 여유 허용)
        except:
            return False

    def _get_processed_image(
        self, url: str, max_dimension: Optional[int] = None
    ) -> Optional[Image.Image]:
        """Downloads, preprocesses, and optionally removes background from an image."""
        img_content = None
        try:
            # Download image with timeout
            response = requests.get(url, timeout=10) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            img_content = response.content

        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout downloading image: {url}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to download image {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading image {url}: {str(e)}", exc_info=True)
            return None

        if not img_content:
            return None

        try:
            img = Image.open(BytesIO(img_content)).convert("RGB")

            # Resize if needed
            img = self._resize_image(img, max_dimension or self.max_image_dimension)

            # Remove background if configured
            if self.remove_background:
                try:
                    img = self._remove_background(img)
                except Exception as bg_err:
                     self.logger.warning(f"Failed to remove background for {url}: {bg_err}")
                     # Continue with original image if background removal fails

            return img

        except UnidentifiedImageError:
            self.logger.warning(f"Cannot identify image file from {url}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing image data from {url}: {str(e)}", exc_info=True)
            return None

    def _resize_image(self, img: Image.Image, max_dimension: Optional[int] = None) -> Image.Image:
        """이미지 크기 조정 (메모리 사용량과 계산 시간 최적화)"""
        try:
            if not max_dimension:
                return img
                
            # 원본 크기
            width, height = img.size
            
            # 이미 충분히 작은 이미지면 그대로 반환
            if width <= max_dimension and height <= max_dimension:
                return img
                
            # 비율 유지하며 크기 조정
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
                
            # 크기 조정 시 안티앨리어싱 적용
            return img.resize((new_width, new_height), Image.LANCZOS)
        except Exception as e:
            self.logger.error(f"Error resizing image: {e}")
            return img  # 오류 시 원본 반환

    def _get_hash_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates hash similarity, potentially using cached hashes."""
        # 여러 해시 알고리즘 사용 (평균 해시, 차이 해시, 웨이블릿 해시)
        hash1_avg = self._get_image_hash(img1, "average")
        hash2_avg = self._get_image_hash(img2, "average")

        hash1_phash = self._get_image_hash(img1, "phash")
        hash2_phash = self._get_image_hash(img2, "phash")

        hash1_whash = self._get_image_hash(img1, "whash")
        hash2_whash = self._get_image_hash(img2, "whash")

        if (
            hash1_avg is None
            or hash2_avg is None
            or hash1_phash is None
            or hash2_phash is None
            or hash1_whash is None
            or hash2_whash is None
        ):
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

    def _get_image_hash(
        self, img: Image.Image, hash_type: str = "average"
    ) -> Optional[imagehash.ImageHash]:
        """Gets the hash of an image using specified algorithm, using cache if available."""
        try:
            if hash_type == "average":
                img_hash = imagehash.average_hash(img)
            elif hash_type == "phash":
                img_hash = imagehash.phash(img)
            elif hash_type == "whash":
                img_hash = imagehash.whash(img)
            else:
                img_hash = imagehash.average_hash(img)
            return img_hash
        except Exception as e:
            self.logger.warning(f"Could not calculate image hash: {e}")
            return None

    def _get_feature_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates feature similarity, potentially using cached features."""
        # Ensure models are loaded via property access
        if self.model is None:
             self.logger.error("EfficientNet model is not available for feature similarity.")
             return 0.0

        feat1 = self._get_image_features(img1)
        feat2 = self._get_image_features(img2)

        if feat1 is None or feat2 is None:
            return 0.0

        similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
        return float(similarity.cpu().numpy())

    def _get_image_features(self, img: Image.Image) -> Optional[torch.Tensor]:
        """Extracts deep features from an image, using cache if available."""
        # Ensure model is loaded via property access
        if self.model is None:
            self.logger.error("EfficientNet model is not available for feature extraction.")
            return None

        try:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.extract_features(img_tensor)
                features = torch.mean(features, dim=[2, 3])  # Global average pooling
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}", exc_info=True)
            return None

    def _get_color_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """두 이미지의 색상 분포 유사도 계산"""
        try:
            # PIL 이미지를 OpenCV 형식으로 변환
            img1_cv = np.array(img1.convert("RGB"))
            img1_cv = img1_cv[:, :, ::-1].copy()  # RGB -> BGR

            img2_cv = np.array(img2.convert("RGB"))
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
        """Calculates similarity based on ResNet features (backup)."""
         # Ensure model is loaded via property access
        if self.backup_model is None:
             self.logger.warning("ResNet backup model is not available.")
             return 0.0

        feat1 = self._get_backup_features(img1)
        feat2 = self._get_backup_features(img2)

        if feat1 is None or feat2 is None:
            return 0.0

        similarity = torch.nn.functional.cosine_similarity(feat1, feat2, dim=0).item()
        return max(0.0, float(similarity))

    def _get_backup_features(self, img: Image.Image) -> Optional[torch.Tensor]:
        """Extracts features using the backup ResNet model."""
         # Ensure model is loaded via property access
        if self.backup_model is None:
            self.logger.warning("ResNet backup model is not available for feature extraction.")
            return None
        try:
            # Use the same transform as the primary model for consistency? Or ResNet specific?
            # Using same transform for now.
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # ResNet feature extraction logic might differ slightly
                # Typically use penultimate layer output
                x = self.backup_model.conv1(img_tensor)
                x = self.backup_model.bn1(x)
                x = self.backup_model.relu(x)
                x = self.backup_model.maxpool(x)

                x = self.backup_model.layer1(x)
                x = self.backup_model.layer2(x)
                x = self.backup_model.layer3(x)
                x = self.backup_model.layer4(x)

                x = self.backup_model.avgpool(x)
                features = torch.flatten(x, 1)
                features = features.squeeze()
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features with ResNet: {e}", exc_info=True)
            return None

    def _remove_background(self, img: Image.Image) -> Image.Image:
        """이미지 배경 제거"""
        try:
            # rembg 사용 시 RGB 모드로 변환 (RGBA 입력 필요)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
                
            output = remove(img)
            
            # 후처리: 투명 배경을 흰색으로 변환 (필요한 경우)
            # bg_removed = Image.new(\"RGBA\", output.size, (255, 255, 255, 255))
            # bg_removed.paste(output, mask=output)
            # return bg_removed.convert('RGB') 
            
            return output.convert('RGB') # rembg는 RGBA 반환, RGB로 변환
        except Exception as e:
            self.logger.error(f"Error removing background: {e}", exc_info=True)
            return img.convert('RGB') if img.mode != 'RGB' else img

    # --- BaseMatcher 추상 메서드 구현 ---

    def find_matches(
        self,
        source_product: Product,
        candidate_products: List[Product],
        min_text_similarity: float = None,
        min_image_similarity: float = None,
        min_combined_similarity: float = None,
        max_matches: int = None,
    ) -> List[MatchResult]:
        """Find best matches among candidates based on image similarity."""
        # TODO: Implement image-based matching logic
        self.logger.warning("find_matches method is not fully implemented in ImageMatcher.")
        
        results = []
        if not source_product.image_url:
            self.logger.debug(f"Source product {source_product.id} has no image URL. Skipping image matching.")
            return []
            
        for target_product in candidate_products:
            if not target_product.image_url:
                self.logger.debug(f"Target product {target_product.id} has no image URL. Skipping.")
                continue

            similarity_score = self.calculate_similarity(
                source_product.image_url, 
                target_product.image_url, 
                self.max_image_dimension
            )

            if min_image_similarity is None:
                min_image_similarity = self.image_similarity_threshold

            if similarity_score >= min_image_similarity:
                results.append(MatchResult(
                    source_product=source_product,
                    matched_product=target_product,
                    similarity_score=similarity_score,
                    match_type="image" 
                ))

        # Sort results by similarity
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Limit results if max_matches is set
        if max_matches is not None:
            results = results[:max_matches]
            
        return results

    def find_best_match(
        self,
        source_product: Product,
        candidate_products: List[Product],
        min_text_similarity: float = None,
        min_image_similarity: float = None,
        min_combined_similarity: float = None,
    ) -> Optional[MatchResult]:
        """Find the single best match based on image similarity."""
        # TODO: Optimize this if possible (e.g., avoid recalculating all matches)
        self.logger.warning("find_best_match method relies on find_matches in ImageMatcher.")
        
        matches = self.find_matches(
            source_product=source_product,
            candidate_products=candidate_products,
            min_image_similarity=min_image_similarity,
            max_matches=1 # We only need the best one
        )
        
        return matches[0] if matches else None

    def batch_find_matches(
        self,
        query_products: List[Product],
        candidate_products: List[Product],
        max_results_per_query: int = None,
        min_similarity: float = None,
    ) -> Dict[str, List[Tuple[Product, float]]]:
        """Batch find image matches for multiple products."""
        # TODO: Implement efficient batch processing
        self.logger.warning("batch_find_matches method is not fully implemented efficiently in ImageMatcher.")
        
        results = {}
        for query_product in query_products:
            # Use find_matches for each query product (can be optimized)
            matches = self.find_matches(
                source_product=query_product,
                candidate_products=candidate_products,
                min_image_similarity=min_similarity, 
                max_matches=max_results_per_query
            )
            
            # Format results as required: Dict[str, List[Tuple[Product, float]]]
            results[query_product.id] = [(match.matched_product, match.similarity_score) for match in matches]
            
        return results
