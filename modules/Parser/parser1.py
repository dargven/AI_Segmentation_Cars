######### LOW QUALITY BUT FASTEST!


import random
import time
import base64

import requests
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()


# def agree_with_google():

#     ## Применяется, когда вылазит надоедливое окошко(при использовании хром). Т.к окошко вылазит
#     ## с определенной переодичностью и не встречается на Windows, реализация функционала
#     ## оставлено на комментировании в случаях, когда окошко перестает вылазить.
#
#     driver.get('https://google.com/xhtml')
#     time.sleep(2)  # seconds until popup appears
#     try:  # 2 different popups
#         frame = driver.find_element(By.XPATH, '//*[@id="cnsw"]/iframe')  # <-locating chrome cookies consent frame
#         if frame is not None:
#             driver.switch_to.frame(frame)
#             button = driver.find_element(By.XPATH, '//*[@id="introAgreeButton"]')
#             if button:
#                 button.click()
#     except NoSuchElementException:
#         button = driver.find_element(By.XPATH, '//*[@id="L2AGLb"]')
#         if button:
#             button.click()
# agree_with_google()


def saved_images_to_download(images, count, filtered_images):
    for image in images:
        src = image.get_attribute('src')
        if (src is not None) and (src not in filtered_images):
            filtered_images.append(src)
            if len(filtered_images) >= count:
                break
        else:
            continue


def search_images(count=1000):
    filtered_images = []
    search_term = "car side view"
    url = (f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch&sa=X&ved"
           f"=2ahUKEwie44_AnqHpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947")
    driver.get(url)
    wait = WebDriverWait(driver, 1)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'rg_i')))
    while len(filtered_images) < count:
        time.sleep(2)
        images = driver.find_elements(By.CLASS_NAME, 'rg_i')
        saved_images_to_download(images, count, filtered_images)
        time.sleep(4)
        scroll_page()
    return filtered_images


def scroll_page():
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)


def download_images(images_urls):
    save_images = 0
    for url in images_urls:
        if url.startswith('data:image'):
            head, data = url.split(',', 1)
            file_ext = '.' + head.split(';')[0].split('/')[1]
            plain_data = base64.b64decode(data)
            save_path = f'src/images/image{save_images}{file_ext}'
            with open(save_path, "wb") as file:
                file.write(plain_data)
                save_images += 1
        else:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    file_ext = '.' + content_type.split('/')[-1]
                    save_path = f'../../src/parsImages/image{save_images}{file_ext}'
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                        save_images += 1
            except Exception as e:
                print(f"Error downloading image from {url}: {e}")


image_links = search_images()
download_images(image_links)
