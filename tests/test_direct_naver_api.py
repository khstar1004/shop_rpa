#!/usr/bin/env python3
"""
네이버 API 직접 테스트 스크립트
API 인증 및 요청이 제대로 동작하는지 확인하기 위한 테스트입니다.
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 확인
client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")

if not client_id or not client_secret:
    print("Error: API 키를 찾을 수 없습니다. .env 파일에 client_id와 client_secret이 설정되어 있는지 확인하세요.")
    sys.exit(1)

print(f"API 키 확인: {client_id[:4]}... / {client_secret[:4]}...")

# API 정보 설정
api_url = "https://openapi.naver.com/v1/search/shop.json"
headers = {
    "X-Naver-Client-Id": client_id,
    "X-Naver-Client-Secret": client_secret,
    "Accept": "application/json"
}

# 검색어 및 파라미터 설정
search_query = "텀블러"
params = {
    "query": search_query,
    "display": 10,  # 결과 개수
    "start": 1,     # 시작 위치 
    "sort": "sim"   # 정렬 (sim: 정확도순, date: 날짜순, asc: 가격오름차순, dsc: 가격내림차순)
}

print(f"\n[요청 정보]")
print(f"URL: {api_url}")
print(f"Headers: {json.dumps(headers, indent=2, ensure_ascii=False)}")
print(f"Parameters: {json.dumps(params, indent=2, ensure_ascii=False)}")

try:
    # API 호출
    print("\n[API 요청 중...]")
    response = requests.get(api_url, headers=headers, params=params, timeout=10)
    
    # 응답 상태 코드 확인
    print(f"\n[응답 정보]")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"총 검색 결과: {data.get('total', 0)}개")
        print(f"현재 페이지 아이템 수: {len(data.get('items', []))}개")
        
        # 결과 출력
        if data.get('items'):
            print("\n[검색 결과]")
            for i, item in enumerate(data['items'][:5], 1):
                title = item.get('title', '').replace('<b>', '').replace('</b>', '')
                price = item.get('lprice', '0')
                mall = item.get('mallName', '')
                
                print(f"{i}. {title}")
                print(f"   가격: {price}원")
                print(f"   쇼핑몰: {mall}")
                print(f"   링크: {item.get('link', '')}")
                print("---")
        else:
            print("검색 결과가 없습니다.")
    else:
        print(f"API 요청 실패: {response.status_code}")
        print(f"응답 내용: {response.text}")
        
except Exception as e:
    print(f"오류 발생: {str(e)}")
    sys.exit(1) 