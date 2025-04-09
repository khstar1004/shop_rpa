#!/usr/bin/env python3
"""
네이버 API 이미지 URL 포함 테스트 스크립트
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# API 키 확인
client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")

if not client_id or not client_secret:
    print("Error: API 키를 찾을 수 없습니다. .env 파일 확인 필요")
    sys.exit(1)

# API 요청 설정
api_url = "https://openapi.naver.com/v1/search/shop.json"
headers = {
    "X-Naver-Client-Id": client_id,
    "X-Naver-Client-Secret": client_secret
}
params = {
    "query": "쿠로미 열냉각시트 6매입",
    "display": 10,
    "sort": "sim"
}

try:
    response = requests.get(api_url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"[검색 결과 총 {data['total']}개]")
        
        # 이미지 URL 포함 출력
        for i, item in enumerate(data['items'][:5], 1):
            title = item['title'].replace('<b>', '').replace('</b>', '')
            print(f"\n{i}. {title}")
            print(f"   가격: {item['lprice']}원")
            print(f"   쇼핑몰: {item['mallName']}")
            print(f"   이미지 URL: {item['image']}")  # 이미지 URL 추가
            print(f"   상세 링크: {item['link']}")
            
    else:
        print(f"API 오류: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"실행 오류: {str(e)}")
