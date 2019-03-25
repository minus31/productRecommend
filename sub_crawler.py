from selenium import webdriver


def login(driver, login_info):
    url = "https://www.musinsa.com/"

    driver.find_element_by_css_selector(
        "#wrapper > div.top-column.column > div > div.userMenu-wrapper.toggleBox > ul > li.listItem.loginBtn > a").click()

    driver.find_element_by_name("id").send_keys(login_info["id"])
    driver.find_element_by_name("pw").send_keys(login_info["pw"])

    driver.find_element_by_css_selector(
        'body > div.bottom-column.column.clearfix > div > div.loginBoxV3 > form > span.submit.submitWBOX > input[type="submit"]').click()

    return True


def get_item_img(spec_url, login_info):
    driver = webdriver.Chrome()
    if login_info:
        login(driver, login_info)

    driver.get(spec_url)

    related_imgs_urls = []

    related_imgs = driver.find_elements_by_css_selector(
        "div .storeListBox > ul > li.listItem > div.articleImg")

    for img in related_imgs:

        related_imgs_url = img.find_element_by_css_selector(
            "a").get_attribute('href')
        related_imgs_urls.append(related_imgs_url)

    return related_imgs_urls