import os
import re
import win32com.client as win32
from PIL import Image as PILImage

def insert_and_anchor_image_to_open_excel(image_path, target):
    # 절대 경로 확인
    abs_image_path = os.path.abspath(image_path)

    # 이미지 파일 존재 여부 검사
    if not os.path.exists(abs_image_path):
        print(f"경로 {abs_image_path}가 존재하지 않습니다.")
        return

    # 이미지 크기를 100x100으로 조정
    with PILImage.open(abs_image_path) as img:
        img = img.convert("RGB") # 투명도 제거
        img_resized = img.resize((100, 100))
        abs_image_path = os.path.splitext(abs_image_path)[0] + "_resized" + os.path.splitext(abs_image_path)[1]
        img_resized.save(abs_image_path)
    
    # 이미 열려있는 Excel 애플리케이션 객체 연결
    excel = win32.GetActiveObject("Excel.Application")
    
    # 활성화된 워크북 가져오기
    workbook = excel.ActiveWorkbook

    # target이 유효한 셀 주소인지 확인
    if not re.match("^[A-Z]+[0-9]+$", target):
        print(f"유효하지 않은 셀 주소 {target}.")
        return

    worksheet = workbook.Worksheets(1)  # 첫 번째 워크시트

    # 셀 주소에서 행과 열 추출
    col_letter = re.match("[A-Z]+", target).group()
    row_number = int(re.search("\d+", target).group())

    # 셀의 크기를 조정된 이미지의 크기에 맞춤
    POINTS_IN_INCH = 72
    PIXELS_PER_POINT = 96 / POINTS_IN_INCH  # 가정: 화면은 96 DPI

    worksheet.Cells(row_number, col_letter).ColumnWidth = 100 / 6.25  # 8은 대략적인 변환 값
    worksheet.Cells(row_number, col_letter).RowHeight = 100

    # 이미지 삽입
    start_cell = worksheet.Range(target)
    left = start_cell.Left
    top = start_cell.Top

    worksheet.Shapes.AddPicture(abs_image_path, LinkToFile=False, SaveWithDocument=True, Left=left, Top=top, Width=100, Height=100)

    # 워크북 저장
    workbook.Save()
