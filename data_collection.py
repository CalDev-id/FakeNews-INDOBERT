import threading
from selenium import webdriver
import time 

class DataCollection():
    def __init__(self, lang, num_pages):
        self.threadLocal = threading.local()
        self.lang = lang
        self.num_pages = num_pages

    def get_driver(self):
        driver = getattr(self.threadLocal, 'driver', None)
        #check browser sudah run atau belum
        if driver is None:
            chromeOptions = webdriver.ChromeOptions()
            chromeOptions.add_argument('--headless=new')
            driver = webdriver.Chrome(options=chromeOptions)



            setattr(self.threadLocal, 'driver', driver)

        return driver
    
    def search(self, query, lang):
        for i_p in range(self.pages):
            start_index = i_p * 10
            url = "https://www.google.com/search"\
            f"q={query}&"\
            f"hl={self.lang}&"\
            f"lr={self.lang}&"\
            f"start={start_index}"

    def fetch_search_result(self, url):
        driver = self.get_driver()
        driver.implicity_wait(10)
        driver.set_page_load_timeout(10)

        driver.get(url)

        time.sleep(5)
        page_content = driver.page_source
        print(page_content)

if __name__ == "__main__":
    data_collection = DataCollection(lang="id", num_pages=10)
    data_collection.search("ibukota indonesia", "id")