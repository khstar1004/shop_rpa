import cv2
import numpy as np
import os

def sift_and_akaze_image_matching(main_folder_path, target_folder_path):
    # SIFT 객체 생성
    sift = cv2.SIFT_create()
    # AKAZE descriptor 초기화
    akaze = cv2.AKAZE_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ransac_reproj_threshold = 5.0
    ransac_max_trials = 1000
    match_param = 50

    main_images = os.listdir(main_folder_path)
    target_images = os.listdir(target_folder_path)

    matching_images_sift = set()
    matching_images_akaze = set()

    for main_image in main_images:
        main_path = os.path.join(main_folder_path, main_image)
        main_img = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE)

        try:
            if main_img is None:
                raise Exception(f"Main image {main_image} is None.")

            if main_img.dtype != np.uint8:
                raise Exception(f"Main image {main_image} has an invalid dtype.")

            main_kp_sift, main_des_sift = sift.detectAndCompute(main_img, None)
            main_kp_akaze, main_des_akaze = akaze.detectAndCompute(main_img, None)

            if main_des_sift is None or main_des_akaze is None:
                raise Exception(f"Main image {main_image} descriptor is None.")
        except Exception as e:
            print(f"Error in processing main image {main_image}: {e}")
            continue

        for target_image in target_images:
            target_path = os.path.join(target_folder_path, target_image)
            target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

            try:
                if target_img is None:
                    raise Exception(f"Target image {target_image} is None.")

                if target_img.dtype != np.uint8:
                    raise Exception(f"Target image {target_image} has an invalid dtype.")

                target_kp_sift, target_des_sift = sift.detectAndCompute(target_img, None)
                target_kp_akaze, target_des_akaze = akaze.detectAndCompute(target_img, None)

                if target_des_sift is None or target_des_akaze is None:
                    raise Exception(f"Target image {target_image} descriptor is None.")
            except Exception as e:
                print(f"Error in processing target image {target_image}: {e}")
                continue

            # SIFT 이미지매칭
            try:
                matches_sift = flann.knnMatch(main_des_sift, target_des_sift, k=2)
                good_matches_sift = [m for m, n in matches_sift if m.distance < 0.87 * n.distance]
            except cv2.error as e:
                print(f"Error in SIFT matching for target image {target_image}: {e}")
                continue

            # AKAZE 이미지매칭
            try:
                matches_akaze = bf.match(main_des_akaze, target_des_akaze)
                matches_akaze = sorted(matches_akaze, key=lambda x: x.distance)
            except cv2.error as e:
                print(f"Error in AKAZE matching for target image {target_image}: {e}")
                continue

            # 변수
            if len(good_matches_sift) > 3:
                filename, file_extension = os.path.splitext(target_image)
                matching_images_sift.add(filename)

            if len(matches_akaze) > 4 or matches_akaze[0].distance < match_param:
                filename, file_extension = os.path.splitext(target_image)
                matching_images_akaze.add(filename)

    # 두 결과값 합집합
    intersection = matching_images_sift or matching_images_akaze

    return ",".join(intersection)

if __name__ == "__main__":
    main_folder_path = "../RPA/Image/Main"
    target_folder_path = "../RPA/Image/Target"

    matches = sift_and_akaze_image_matching(main_folder_path, target_folder_path)

    if matches:
        print("두 방법 모두에서 매칭된 이미지가 발견되었습니다:")
        print(matches)
    else:
        print("두 방법 모두에서 매칭된 이미지가 없습니다.")
