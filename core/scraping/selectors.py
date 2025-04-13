"""
고려기프트 웹사이트 스크래핑을 위한 셀렉터 정의
"""

KORYO_SELECTORS = {
    # 메인 페이지 셀렉터
    "main_page": {
        "product_list": {
            "selector": ".best100_tab .product, .prd_list_wrap li.prd, ul.prd_list li, table.mall_list td.prd, div.product_list .item, .prd-list .prd-item"
        },
        "product_title_list": {
            "selector": ".name, p.name a, div.name a, td.name a, .prd_name a, a.product-title, div.name > a"
        },
        "product_link_list": {
            "selector": "a, p.name a, div.name a, td.name a, .prd_name a, a.product-title, div.prd_list_wrap li.prd > a, div.pic > a",
            "attribute": "href"
        },
        "price_list": {
            "selector": ".price, p.price, div.price, td.price, .prd_price, span.price"
        },
        "thumbnail_list": {
            "selector": ".img img, .pic img, img.prd_img, td.img img, .thumb img, img.product-image, div.pic > a > img",
            "attribute": "src"
        },
        "next_page_js": {
            "selector": '#pageindex a[onclick*="getPageGo"], div.custom_paging div[onclick*="getPageGo"], .paging a.next, a:contains("다음")'
        },
        "next_page_href": {
            "selector": '#pageindex a:not([onclick]), .paging a.next, a.next[href]:not([href="#"]):not([href^="javascript:"]), a:contains("다음")[href]:not([href="#"]):not([href^="javascript:"])',
            "attribute": "href"
        },
        "category_items": {
            "selector": "#div_category_all .tc_link, #lnb_menu > li > a, .category a, #category_all a, .menu_box a, a[href*='mall.php?cat='], a[href*='mall.php?cate=']"
        },
        "main_search_input": {
            "selector": 'input[name="keyword"][id="main_keyword"]'
        },
        "main_search_button": {
            "selector": 'img#search_submit'
        }
    },
    
    # 상품 상세 페이지 셀렉터
    "product_page": {
        "product_name": {
            "selector": ".product_name"
        },
        "product_code": {
            "selector": "div:contains('상품코드:')"
        },
        "main_image": {
            "selector": "#main_img",
            "attribute": "src"
        },
        "thumbnail_images": {
            "selector": ".product_picture .thumnails img",
            "attribute": "src"
        },
        "quantity_table": {
            "selector": "table.quantity_price__table"
        },
        "specs_table": {
            "selector": "table.tbl_info"
        },
        "description": {
            "selector": "div.prd_detail, #prd_detail_content"
        },
        "options": {
            "selector": 'select[name^="option_"]'
        },
        "price": {
            "selector": "#main_price"
        },
        "total_price": {
            "selector": "#z_hap"
        },
        "final_price": {
            "selector": "#z_hap2"
        },
        "delivery_cost": {
            "selector": "#delivery_span"
        },
        "review_table": {
            "selector": "table.tbl_review"
        },
        "review_pagination": {
            "selector": "#pageindex"
        }
    }
} 