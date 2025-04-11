import csv
from sentence_transformers import SentenceTransformer, util

# 경로 변수 설정
INPUT_PATH = "C:\\RPA\\Image\\Target\\input.csv"
OUTPUT_PATH = "C:\\RPA\\Image\\Target\\output.csv"

def calculate_similarity():
    # Ko-Sentence-BERT 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # CSV 파일에서 문장 리스트 불러오기
    with open(INPUT_PATH, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]

    # 첫 번째 라인을 메인 문장으로 설정하고 임베딩 계산
    main_sentence = sentences[0]
    embedding_main = model.encode(main_sentence, convert_to_tensor=True)

    similarities = []

    # 나머지 라인들과의 유사도 계산
    for sentence in sentences[1:]:
        embedding_sentence = model.encode(sentence, convert_to_tensor=True)
        
        cosine_sim = util.pytorch_cos_sim(embedding_main, embedding_sentence)
        similarities.append(cosine_sim.item())

    # 유사도를 CSV 파일에 저장
    save_to_csv(similarities)

def save_to_csv(data):
    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for value in data:
            writer.writerow([value])