import pandas as pd
from playwright.sync_api import sync_playwright
import time # time 모듈 임포트
import random # random 모듈 임포트
import logging # logging 모듈 임포트

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 일반적인 사용자 에이전트
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"

def scrape_data(keyword1, keyword2=None, output_path=r'C:\RPA\product_lists.xlsx'):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True) # Headless 모드 활성화
            context = browser.new_context(user_agent=USER_AGENT) # 사용자 에이전트 설정
            page = context.new_page()
            page.goto("https://koreagift.com/ez/index.php")
            logging.info("Accessed koreagift.com")

            time.sleep(random.uniform(1, 3)) # 초기 로드 대기

            page.locator('//input[@name="keyword" and @id="main_keyword"]').fill(keyword1)
            page.locator('//img[@id="search_submit"]').click()
            page.wait_for_load_state('networkidle', timeout=60000) # 타임아웃 증가
            logging.info(f"Initial search completed for keyword: {keyword1}")

            time.sleep(random.uniform(1, 3)) # 검색 후 대기

            # 첫 검색 상품 개수
            try:
                product_count_element = page.locator('//div[text()=" 개의 상품이 있습니다."]/span')
                productCont = product_count_element.text_content(timeout=10000) # 타임아웃 설정
                productCont = int(productCont.replace(',', ''))
                logging.info(f"Found {productCont} products for initial search.")
            except Exception as e:
                logging.warning(f"Could not find product count after initial search: {e}")
                productCont = 0

            if keyword2 and keyword2.strip() != "" and productCont >= 100:
                logging.info(f"Initial product count ({productCont}) >= 100. Performing re-search with keyword: {keyword2}")
                time.sleep(random.uniform(1, 3)) # 재검색 전 대기
                page.locator('//input[@id="re_keyword"]').fill(keyword2)
                page.locator('//button[@onclick="re_search()"]').click()
                page.wait_for_load_state('networkidle', timeout=60000) # 타임아웃 증가
                logging.info(f"Re-search completed for keyword: {keyword2}")

                time.sleep(random.uniform(1, 3)) # 재검색 후 대기
                # 재검색 상품 개수
                try:
                    product_count_element = page.locator('//div[text()=" 개의 상품이 있습니다."]/span')
                    productCont = product_count_element.text_content(timeout=10000) # 타임아웃 설정
                    productCont = int(productCont.replace(',', ''))
                    logging.info(f"Found {productCont} products after re-search.")
                except Exception as e:
                    logging.warning(f"Could not find product count after re-search: {e}")
                    productCont = 0

            data = []
            page_number = 1

            if productCont > 0:
                while True:
                    logging.info(f"Scraping page {page_number}...")
                    time.sleep(random.uniform(0.5, 1.5)) # 페이지 처리 전 짧은 대기

                    # 각 페이지의 제품 데이터를 가져옵니다.
                    rows = page.locator('//div[@class="product"]')
                    count = rows.count()
                    logging.info(f"Found {count} products on page {page_number}.")

                    for i in range(count):
                        row = rows.nth(i)
                        try:
                            img_src = row.locator('div.pic > a > img').get_attribute('src', timeout=5000)
                            total_src = "http://koreagift.com/ez/" + img_src.replace("./", "") if img_src else ""

                            a_href = row.locator('div.pic > a').get_attribute('href', timeout=5000)
                            total_href = "http://koreagift.com" + a_href if a_href else ""

                            name = row.locator('div.name > a').text_content(timeout=5000)
                            price_text = row.locator('div.price').text_content(timeout=5000)
                            price = price_text.replace('원', '').replace(',', '') if price_text else ""

                            data.append({
                                'name': name.strip() if name else "",
                                'href': total_href.strip(),
                                'src': total_src.strip(),
                                'price': price
                            })
                        except Exception as e:
                            logging.warning(f"Could not extract data for an item on page {page_number}, item index {i}: {e}")
                            continue # 문제가 있는 항목은 건너뛰기

                    # 다음 페이지로 이동
                    next_page_selector = f'//div[@class="custom_paging"]/div[@onclick="getPageGo1({page_number + 1})"]'
                    next_page = page.locator(next_page_selector)

                    if next_page.count() == 0:
                        logging.info("No more pages found. Scraping finished.")
                        break

                    time.sleep(random.uniform(1, 3)) # 다음 페이지 클릭 전 대기
                    try:
                        next_page.click(timeout=10000) # 타임아웃 설정
                        page.wait_for_load_state('networkidle', timeout=60000) # 타임아웃 증가
                        page_number += 1
                    except Exception as e:
                        logging.warning(f"Failed to navigate to next page ({page_number + 1}): {e}")
                        break # 페이지 이동 실패 시 종료

            browser.close()

            if data:
                df = pd.DataFrame(data)
                df.to_excel(output_path, index=False)
                logging.info(f"Successfully scraped {len(data)} items and saved to {output_path}")
                # print(df) # 필요시 주석 해제
            else:
                logging.info("No data was scraped.")

            return len(data) if data else 0

        except Exception as e:
            logging.error(f"An error occurred during scraping: {e}", exc_info=True)
            if 'browser' in locals() and browser.is_connected():
                browser.close()
            return 0 # 오류 발생 시 0 반환


if __name__ == "__main__":
    # 결과 파일 경로 지정 가능
    result_count = scrape_data("송월", "스누피", output_path=r'C:\RPA\haerorm_gift_products.xlsx')
    logging.info(f"Scraping process finished. Total items saved: {result_count}")