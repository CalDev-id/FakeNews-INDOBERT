import sys, os
import requests
import threading
import json
import logging

import re

import pandas as pd

import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# import eventlet

from bs4 import BeautifulSoup
from tqdm import tqdm

from googlesearch import search
import trafilatura

from tools.evidence_ranker import EvidenceRanker

class EvidenceSearch():
    
    def __init__(self,
                 lang,
                 pages = 1,
                 max_query_search = 100,
                 max_content_search = 5,
                 sort_by = "evidence_query"):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36 OPR/58.0.3135.79'}
        self.lang = lang
        self.pages = pages
        self.threadLocal = threading.local()
        
        self.max_query_search = max_query_search
        self.max_content_search = max_content_search
        self.timeout = 10
        
        self.sort_by = sort_by
        
        self.evidence_ranker = EvidenceRanker(selection_max_len = 512)
        
        
    def get_driver(self):
        driver = getattr(self.threadLocal, 'driver', None)

        if driver is None:
            chromeOptions = webdriver.ChromeOptions()
            chromeOptions.add_argument('--headless=new')
            driver = webdriver.Chrome(options=chromeOptions)
            setattr(self.threadLocal, 'driver', driver)
        
        return driver

    
    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"javascript", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()
        
        return string
    
    def visit_content(self, target_url):
        try:
            page_raw = requests.get(target_url, headers = self.headers, timeout = self.timeout, verify=False)
            
            if page_raw.status_code == 400:
                print("timeout")
                print(target_url)
                page_content = {}
            else:
                page_content = trafilatura.bare_extraction(page_raw.text)
                page_content = {
                    "date": page_content["date"],
                    "author": page_content["author"],
                    "text": page_content["text"],
                    "language": page_content["language"],
                    "url": page_content["url"],
                    "hostname": page_content["hostname"],
                }
        except:
            # logging.error("Time out!")
            
            print("timeout")
            print(target_url)
            page_content = {}
        
        
        return page_content
    
    def fetch_content(self, url, claim, query):
        contents_data = []
        
        driver = self.get_driver()
        driver.get(url)
        
        contents = BeautifulSoup(driver.page_source, "html.parser")
        
        search_highlight = ""
        highlights = contents.find_all("span", attrs={"class": "hgKElc"})
        
        # if len(highlights) > 0:
        #     search_highlight = highlights[0].text
        
        for i_cts, cts in enumerate(contents.find_all("div", attrs={'class': 'MjjYud'})):
            title = cts.find_all("h3", attrs={'class': 'DKV0Md'})
            
            if len(title) > 0:
                title = title[0].text
                url = cts.find("cite")
                root_url = url.text.split(" › ")[0]
                source = cts.find_all("span")[0].text
                source_url = cts.find_all("a", attrs={"jsname": "UWckNb"})[0]["href"]
                
                searched_content = self.visit_content(source_url)
                
                if len(searched_content) < 1:
                    continue
                
                evidence_scores = self.evidence_ranker.compute_evidence_score_piece(evidence = searched_content["text"],
                                                                                    claim = claim,
                                                                                    query = query)
                
                meta_data = {
                    "title": title,
                    "root_url": root_url,
                    "source": source,
                    "source_url": source_url,
                    "lang": self.lang,
                    "query": query,
                }
                final_data = {**meta_data, **searched_content, **evidence_scores}
                contents_data.append(final_data)

            
            if i_cts >= self.max_content_search:
                break
        
        if self.sort_by == "claim_evidence":
            contents_data = sorted(contents_data, key=lambda contents_data: contents_data['evidence_claim_score'], reverse = True)
        else:
            contents_data = sorted(contents_data, key=lambda contents_data: contents_data['evidence_query_score'], reverse = True)
            
        return contents_data
            

    def search_piece(self, query, claim):
        evidence = []
        
        for i in range(self.pages):
            start_index = i * 10
            url = f"https://www.google.com/search?" \
                f"q={query}&"\
                f"hl={self.lang}&" \
                f"lr=lang_{self.lang}&" \
                f"start={start_index}"
            
            contents_data = self.fetch_content(url, claim = claim, query = query)
            evidence += contents_data
        
        return evidence

    def search(self, queries, claim, context = None):
        datas = []
        if context:
            claim_context = claim + context
            claim_context = claim_context.strip().split(". ")
            claim_context = ". ".join(claim_context[:5])
        
        for i_query, query in enumerate(queries):
            evidence = self.search_piece(query = query["query"], claim = claim_context)
            
            datas.append({
                "query": query["query"],
                "query_score": query["query_score"],
                "claim": claim,
                "context": context,
                "evidence": evidence
            })    
            if i_query > self.max_query_search:
                break
            
        return datas
    
        
    
    def translations(self, txt_origin, lang_origin, lang_target):
        url = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl={lang_target}&q={txt_origin}"
        response = requests.get(url)
        txt_translated = json.loads(response.text)[0][0]
        return txt_translated.strip()
    

    def verify_claim(self):
        with open('datasets/MMCoVaR/MMCoVaR_News_search_queries_retry.json', 'r') as f_read:
            datasets = json.load(f_read)
        
        overall = []
        
        for data in tqdm(datasets):
            if len(data["queries"]) > 0:
                results = self.search(data["queries"])
                overall.append({
                    "claim": data["claim"],
                    "context": data["context"],
                    "queries": data["queries"],
                    "evidence": results
                })
                
                with open(f"datasets/MMCoVaR/MMCoVaR_News_search_queries_evidence.json", "w") as w_json:
                    json.dump(overall, w_json, indent = 4)   
                    
                # print(overall)
                # print(results[0].keys())
                # print("="*20)
                # sys.exit()

if __name__ == '__main__':
    
    Stool = EvidenceSearch(lang = "en", pages = 1)
    # Stool.search(keyword = "What is the second iteration of CBS' Late Show franchise?")
    Stool.verify_claim()
    
    # test = list(search("covid19", advanced=True, num_results = 10, sleep_interval = 5))
    # print(test[0])