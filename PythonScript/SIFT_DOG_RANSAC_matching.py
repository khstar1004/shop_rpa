import os
import cv2
import numpy as np

def image_matching(main_folder_path, target_folder_path):
    # 폴더 내 이미지 파일만 리스트로
    def get_image_files_from_folder(folder_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  
        image_files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))

        return image_files

    # 이미지에서 DoG (Difference of Gaussian)을 계산
    def calculate_dog(image, ksize1, ksize2):
        blurred1 = cv2.GaussianBlur(image, (ksize1, ksize1), 0)
        blurred2 = cv2.GaussianBlur(image, (ksize2, ksize2), 0)
        dog = blurred1 - blurred2
        return dog

    # 이미지에서 SIFT(Scale-Invariant Feature Transform) 매칭 점수를 계산
    def calculate_sift_matching_score(image1, image2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(image1, None)
        kp2, des2 = sift.detectAndCompute(image2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(good_matches) > 5:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            inlier_matches = [m for i, m in enumerate(good_matches) if matches_mask[i] == 1]
            matching_score = len(inlier_matches) / len(good_matches)
        else:
            matching_score = 0.0

        return matching_score

    # 메인 폴더와 타겟 폴더 내의 이미지 파일들을 가져옵니다.
    main_image_files = get_image_files_from_folder(main_folder_path)
    target_image_files = get_image_files_from_folder(target_folder_path)

    matching_results = set()  # 일치율이 0 이상인 target 이미지 파일 이름 저장을 위한 set
    for main_file in main_image_files:
        try:
            main_image = cv2.imread(main_file, cv2.IMREAD_GRAYSCALE)
            if main_image is None:
                raise Exception(f"이미지를 읽을 수 없습니다: {main_file}")

            for target_file in target_image_files:
                try:
                    target_image = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
                    if target_image is None:
                        raise Exception(f"이미지를 읽을 수 없습니다: {target_file}")

                    ksize1 = 3
                    ksize2 = 11
                    main_dog = calculate_dog(main_image, ksize1, ksize2)
                    target_dog = calculate_dog(target_image, ksize1, ksize2)

                    # 두 이미지 간의 SIFT 매칭 점수를 계산합니다.
                    matching_score = calculate_sift_matching_score(main_dog, target_dog)

                    if matching_score > 0:
                        base_name = os.path.basename(target_file)
                        file_name, _ = os.path.splitext(base_name)
                        matching_results.add(file_name)
                except Exception as target_e:
                    print(f"{target_file} 처리 중 오류 발생: {target_e}")
        except Exception as main_e:
            print(f"{main_file} 처리 중 오류 발생: {main_e}")

    return ",".join(matching_results)

# 함수 사용
matching_files = image_matching("C:/RPA/Image/Main", "C:/RPA/Image/Target")
print(f"대상 이미지: {matching_files}")
