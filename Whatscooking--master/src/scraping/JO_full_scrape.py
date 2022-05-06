import pandas as pd 
import requests
import time
from bs4 import BeautifulSoup
import numpy as np
from JO_scrape_class import cuisineFacile

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# The list of recipe attributes we want to scrape
attribs = ["recipe_name", "ingredients"]
recipe_df = pd.read_csv(r"input/recipe_urls.csv")
temp = pd.DataFrame(columns=attribs)
for i in range(0,len(recipe_df['recipe_urls'])):
    url = recipe_df['recipe_urls'][i]
    recipe_scraper = cuisineFacile(url)
    temp.loc[i] = [getattr(recipe_scraper, attrib)() for attrib in attribs]
    if i % 25 == 0:
        print(f'Step {i} completed')
    
# Put all the data into the same dataframe
temp['recipe_urls'] = recipe_df['recipe_urls']
columns = ['recipe_urls'] + attribs
temp = temp[columns]

JamieOliver_df = temp

JamieOliver_df.to_csv(r"input/JamieOliver_full.csv", index=False)

