import os
import pandas as pd

def check_excel_file():
    input_directory = 'C:\\RPA\\Input'
    
    xlsx_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.xlsx')]
    
    if not xlsx_files:
        print("xlsx 파일이 없습니다.")
        return
    
    file_path = os.path.join(input_directory, xlsx_files[0])
    df = pd.read_excel(file_path)

    # 요청하신 컬럼이 있는지 확인합니다.
    columns_to_add = ['본사 이미지', '고려기프트 이미지', '네이버 이미지']
    need_to_modify = False  # 파일을 수정해야 하는지 확인하는 플래그입니다.
    
    for column in columns_to_add:
        if column not in df.columns:
            df[column] = ''
            need_to_modify = True  # 수정 필요 플래그를 True로 설정합니다.

    if not need_to_modify:
        print("필요한 컬럼이 모두 있습니다. 함수를 종료합니다.")
        return

    # 컬럼명의 앞뒤 공백을 제거합니다.
    df.columns = [col.strip() for col in df.columns]

    # 엑셀 파일에 덮어쓰기를 합니다.
    df.to_excel(file_path, index=False)
    print("작업 완료!")