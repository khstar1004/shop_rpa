from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
import yaml
import os

# 설정 파일 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

try:
    # Chrome 옵션 설정
    options = uc.ChromeOptions()
    options.add_argument('--headless')
    
    # Undetected ChromeDriver 초기화
    driver = uc.Chrome(options=options)
    
    # 테스트 URL 접속
    driver.get('https://www.google.com')
    print("ChromeDriver 테스트 성공!")
    print(f"페이지 제목: {driver.title}")
    
    # WebDriver 종료
    driver.quit()
    
except Exception as e:
    print(f"ChromeDriver 테스트 실패: {str(e)}") 