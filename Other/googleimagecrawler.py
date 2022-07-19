from selenium import webdriver
import numpy as np
from selenium.webdriver.common.by import By
from collections import deque
import re
from selenium.common.exceptions import StaleElementReferenceException
import base64
from tqdm import tqdm
import requests
import random
import sys
import argparse
 


    
class ImageCrawler:
    def __init__(self, width_thresh=80, strategy="FIFO", timeout=5):
        options = webdriver.ChromeOptions()
        options.add_argument("--incognito")
        self.driver = webdriver.Chrome(options=options)
        self.driver.get('https://www.google.com.tw/imghp')
        self.width_thresh = width_thresh
        self.downloaded_images = set()
        self.downloaded_urls = set()
        self.google_image_regex = re.compile(r'https://www\.google\.com\.tw/imgres\?imgurl=')
        self.base64_regex = re.compile(r'data:')
        self.google_crawlable_regex = re.compile(r"https://encrypted-tbn0.gstatic.com/images")
        self.saved_url = []
        self.timeout = timeout
        self.strategy = strategy.upper()
        self.image_queue = deque()
        
        self.headers = {
            "Accept": "image/webp,*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-TW,zh;q=0.8,en-US;q=0.5,en;q=0.3",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
#             "Transfer-encoding": "chunked"
        }
        
    def query(self, query_str, flush=True):
        input_list = self.driver.find_elements(By.TAG_NAME, 'input')
        query_input = []
        for e in input_list:
            if e.get_property('type') == 'text':
                query_input.append(e)
        query_input = query_input[0]
        query_input.clear()
        query_input.send_keys(query_str)
        query_input.submit()
        
        if flush:
            self.image_queue = deque()
    
    def fetch(self):
        image_list = self.driver.find_elements(By.TAG_NAME, 'img')
        image_list = np.array(image_list, dtype=object)
        image_exists = np.array([self.is_elem_exists(img) for img in image_list])
        unattached_images = image_list[~image_exists]
        image_list = image_list[image_exists]
        width_list = [img.size['width'] for img in image_list]
        width_list = np.array(width_list, dtype=object).astype(np.int32)
        image_list = image_list[(~np.isnan(width_list)) & (width_list>self.width_thresh)]
        added_images = []
        for img in image_list:
            if img not in self.downloaded_images:
                self.put_image(img)
                self.downloaded_images.add(img)
                added_images.append(img)
        for img in unattached_images:
            if img in self.downloaded_images:
                self.downloaded_images.remove(img)
        return added_images
    
    def is_elem_exists(self, elem):
        try:
            elem.size
            return True
        except StaleElementReferenceException as e:
            return False
    
    def click_into(self, elem):
        try:
            if not self.is_elem_exists(elem):
                return False
            elem = self.get_parent_arch(elem)
            href = elem.get_attribute("href")
            if href is not None and self.google_crawlable_regex.match(href) is None:
                return False
            elem.click()
            return True
        except Exception as e:
            print(e)
            return False
    def scroll_more(self):
        input_elem = self.driver.find_elements(By.TAG_NAME, 'input')
        input_elem = np.array(input_elem, dtype=object)

        is_bn = [elem.get_attribute('type')=='button' for elem in input_elem]
        is_exists = [self.is_elem_exists(elem) for elem in input_elem]
        is_bn = np.array(is_bn)
        is_exists = np.array(is_exists)
        input_elem = input_elem[is_bn & is_exists]

        elem_width = [elem.size['width'] for elem in input_elem]
        is_large = [w>100 for w in elem_width]
        is_large = np.array(is_large)
        elem_width = np.array(elem_width)

        input_elem = input_elem[is_large]
        elem_width = elem_width[is_large]
        ord = np.argsort(elem_width)
        input_elem = input_elem[ord]
        
        if len(input_elem):
            input_elem[-1].click()
        
        self.fetch()
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def next(self, save_path=None, fn=None):
        if len(self.image_queue) == 0:
            self.scroll_more()
        img = self.take_image()
        if self.click_into(img):
            added_imgs = self.fetch()
            for img in added_imgs:
                href = self.get_parent_arch(img).get_attribute("href")
                if href is not None and self.google_image_regex.match(href) is None:
                    self.image_queue.remove(img)
                    self.downloaded_images.remove(img)
                    added_imgs.remove(img)
                    url = img.get_attribute("src")
                    if url not in self.downloaded_urls:
                        self.downloaded_urls.add(url)
                        self.saved_url.append(url)
                        if save_path is not None:
                            b, ext = self.download(url)
                            self.save(save_path, fn, ext, b)
                    if self.strategy == 'LIFO':
                        added_imgs += self.fetch()
                        for i in range(len(added_imgs)-1):
                            self.image_queue.pop()
                        random.shuffle(added_imgs)
                        self.image_queue.extend(added_imgs)
                    return True
        return False
    
    def crawl(self, n=1, save_path=None, counter=1):
        for i in tqdm(range(n)):
            try:
                self.next(save_path=save_path, fn=str(counter))
                counter += 1
            except Exception as e:
                print(e)
    
    def put_image(self, img):
        return self.image_queue.append(img)
    
    def take_image(self):
        if self.strategy == 'FIFO':
            return self.image_queue.popleft()
        elif self.strategy == 'LIFO':
            return self.image_queue.pop()  
    
    def get_parent_arch(self, elem):
        while elem.tag_name != 'a' and elem.tag_name != 'html':
            elem = elem.find_element(By.XPATH, '..')
        return elem
    
    def parse_base64(self, url):
        idx = url.index('base64,')
        ext = url[5:idx-1].split('/')[-1]
        b = url[idx+7:]
        b = bytes(b, 'utf8')
        b = base64.decodebytes(b)
        return b, ext
    def download_url(self, url):
        s = requests.Session()
        s.headers.update(self.headers)
        try:
            r = s.get(url, timeout=self.timeout)
            b = r.content
            contentType = r.headers["Content-type"]
            ext = contentType.split('/')[-1]
            return b, ext
        except requests.exceptions.Timeout:
            print("Timeout: {}".format(url))
            return b'', ''
        
    def download(self, url):
        if self.base64_regex.match(url):
            return self.parse_base64(url)
        else:
            return self.download_url(url)
    def save(self, path, fn, ext, b):
        if path[-1] != "/":
            path = path + "/"
        with open("{}{}.{}".format(path, fn, ext),"wb") as f:
            f.write(b)
        
    def save_all(self, path, counter=1): 
        for i in tqdm(range(0, len(self.saved_url))):
            b, ext = self.download(self.saved_url[i])
            self.save(path, str(counter), ext, b)
            counter += 1
    
    def skip(self, n=0):
        while len(self.image_queue) < n:
            self.scroll_more()
        self.image_queue = deque()
        
if __name__ == '__main__':
     
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keyword", help = "Searching  keyword.", default="")
    parser.add_argument("-d", "--directory", help = "Directory for downloaded images.", default="./")
    parser.add_argument("-n", "--number", help = "Number of image need to crawl.", type=int, default=10)
    parser.add_argument("-s", "--strategy", help = "FIFO search by keyword, LIFO search via related images.", default="FIFO")
    parser.add_argument("-t", "--timeout", help = "Max second waiting for image download.", type=float)
    parser.add_argument("-c", "--counter", help = "First sequence NO. of file name.", type=int, default=1)
    parser.add_argument("-S", "--skip", help = "Skip first n images.", type=int, default=0)
    args = parser.parse_args()
    
    
    crawler = ImageCrawler(strategy=args.strategy, timeout=args.timeout)
    crawler.query(args.keyword) # enter keyword
    if args.skip > 0:
        crawler.skip(args.skip)  # start from args.skip -st image
    # starting crawing
    crawler.crawl(n=args.number, save_path=args.directory, counter=args.counter)
