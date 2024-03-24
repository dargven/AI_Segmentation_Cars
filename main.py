from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.devtools.v122.network import Cookie
from selenium.common.exceptions import NoSuchElementException
import time
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64

driver = webdriver.Chrome()


def agree_with_google():
    driver.get('https://google.com/xhtml')

    time.sleep(2)  # seconds until popup appears

    try:  # 2 different popups
        frame = driver.find_element(By.XPATH, '//*[@id="cnsw"]/iframe')  # <-locating chrome cookies consent frame
        driver.switch_to.frame(frame)
        driver.find_element(By.XPATH,
                            '//*[@id="introAgreeButton"]').click()  # <-looking for introAgreeButton button, but seems google has changed its name since and  it only works in old chrome versions.

    except NoSuchElementException:
        driver.find_element(By.XPATH, '//*[@id="L2AGLb"]').click()


def search_images():
    search_term = "car side view"
    url = (f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch&sa=X&ved"
           f"=2ahUKEwie44_AnqHpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947")
    driver.get(url)
    wait = WebDriverWait(driver, 5)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'rg_i')))
    elements = (driver.find_elements(By.CLASS_NAME, 'rg_i'))
    return [element.get_attribute("src") for element in elements]


def scroll_page():
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)


def download_images(images_urls):
    save_images = 0

    while save_images < len(images_urls):
        src = images_urls[save_images]
        if src and ',' in src:
            head, data = src.split(',', 1)
            file_ext = '.' + head.split(';')[0].split('/')[1]
            plain_data = base64.b64decode(data)
            save_path = 'images/image'

            with open(save_path + str(save_images) + file_ext, "wb") as file:
                file.write(plain_data)
                save_images += 1


agree_with_google()
image_links = search_images()
download_images(image_links)
