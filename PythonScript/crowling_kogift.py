import pandas as pd
from playwright.sync_api import sync_playwright

def scrape_data(keyword1, keyword2=None):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://koreagift.com/ez/index.php")

        page.locator('//input[@name="keyword" and @id="main_keyword"]').fill(keyword1)
        page.locator('//img[@id="search_submit"]').click()
        page.wait_for_load_state('networkidle')
        
        # 첫 검색 상품 개수
        productCont = page.locator('//div[text()=" 개의 상품이 있습니다."]/span').text_content()
        productCont = int(productCont.replace(',', ''))
        print(productCont)
        
        if keyword2 != "" and productCont >= 100:
            page.locator('//input[@id="re_keyword"]').fill(keyword2)
            page.locator('//button[@onclick="re_search()"]').click()
            page.wait_for_load_state('networkidle')
            # 재검색 상품 개수
            productCont = page.locator('//div[text()=" 개의 상품이 있습니다."]/span').text_content()
            productCont = int(productCont)
            100
            

        data = []
        page_number = 1
        
        if productCont > 0:
            while True:
                print(f"Scraping page {page_number}...")
                
                # 각 페이지의 제품 데이터를 가져옵니다.
                rows = page.locator('//div[@class="product"]')
                count = rows.count()
                
                for i in range(count):
                    row = rows.nth(i)
                    # CSS 선택자를 사용하여 img 태그의 src 속성 값을 가져옵니다.
                    img_src = row.locator('div.pic > a > img').get_attribute('src')
                    total_src = "http://koreagift.com/ez/" + img_src.replace("./", "")
                    # CSS 선택자를 사용하여 a 태그의 href 속성 값을 가져옵니다.
                    a_href = row.locator('div.pic > a').get_attribute('href')
                    total_href = "http://koreagift.com" + a_href
                    # CSS 선택자를 사용하여 제품 이름과 가격을 가져옵니다.
                    name = row.locator('div.name > a').text_content()
                    price = row.locator('div.price').text_content()
                    price = price.replace('원', '').replace(',', '')
                    data.append({
                        'name': name,
                        'href': total_href.strip(),
                        'src': total_src.strip(),
                        'price': price
                    })
                
                # 다음 페이지로 이동
                next_page_selector = f'//div[@class="custom_paging"]/div[@onclick="getPageGo1({page_number + 1})"]'
                # next_page_selector = f'//div[@class="tbl_paging"]/a[span[text()="{page_number + 1}"]]'
                next_page = page.locator(next_page_selector)
                
                if next_page.count() == 0:
                    # 다음 페이지가 없으면 종료
                    break
                
                next_page.click()
                page.wait_for_load_state('networkidle')
                page_number += 1
            
            browser.close()
            
            df = pd.DataFrame(data)
            
            df.to_excel(r'C:\RPA\product_lists.xlsx', index=False)
            
            print(df)

        return productCont
        

if __name__ == "__main__":
    scrape_data("송월", "스누피")