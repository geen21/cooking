import requests
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import re

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"}
class cuisineFacile():
    def __init__(self, url):
        self.url = url
        self.soup = BeautifulSoup(requests.get(url, verify=False).content, 'html.parser')
    def recipe_name(self):
        try:
            return self.soup.find('h1').text.strip()
        except: 
            return np.nan
    def ingredients(self):
        try:
            ingredients = []
            for li in self.soup.select(".list-group-item.pt-2.pb-2"):
                ingred = " ".join(li.text.split())
                ingredients.append(ingred)
            return ingredients
        except:
            return np.nan
