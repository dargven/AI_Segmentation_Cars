from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

search_term = "car side view"
url = (f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch&sa=X&ved"
       f"=2ahUKEwie44_AnqHpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947")

driver = webdriver.Firefox()
driver.get(url)
wait = WebDriverWait(driver, 5)
wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'rg_i')))
elements = (driver.find_elements(By.CLASS_NAME, 'rg_i'))
image_links = [element.get_attribute("src") for element in elements]