from selenium import webdriver


def login(driver, login_info):
    url = "https://www.musinsa.com/"
    driver.get(url)

    driver.find_element_by_css_selector(
        "#wrapper > div.top-column.column > div > div.userMenu-wrapper.toggleBox > ul > li.listItem.loginBtn > a").click()

    driver.find_element_by_name("id").send_keys(login_info["id"])
    driver.find_element_by_name("pw").send_keys(login_info["pw"])

    driver.find_element_by_css_selector(
        'body > div.bottom-column.column.clearfix > div > div.loginBoxV3 > form > span.submit.submitWBOX > input[type="submit"]').click()

    return True


def get_item_img(spec_url, login_info=None, close=False):
    driver = webdriver.Chrome()
    if login_info:
        login(driver, login_info)

    driver.get(spec_url)

    related_items_urls = []

    related_items = driver.find_elements_by_css_selector(
        r"div > ul.styleItem-list > li.listItem")

    for block in related_items:
        related_items = block.find_elements_by_css_selector("div.articleImg")
        block_imgs = []
        for item in related_items:

            item_url = item.find_element_by_css_selector(
                "a").get_attribute('href')
            block_imgs.append(item_url)

        related_items_urls.append(block_imgs)
        
    if close:
        driver.close()

    return related_items_urls