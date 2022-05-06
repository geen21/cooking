import requests
from bs4 import BeautifulSoup
import pandas as pd

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://cuisine-facile.com/liste.php"
page = requests.get(url, verify=False)
soup = BeautifulSoup(page.text, "html.parser")

# Initializing DataFrame to store the scraped URLs
recipe_url_df = pd.DataFrame() 
recipe_urls = pd.Series([a.get("href") for a in soup.find_all("a")])
recipe_urls = recipe_urls[(recipe_urls.str.contains("recette-")==True) & (recipe_urls.str.contains("php")==True) ].unique()

# BeautifulSoup enables to find the elements/tags in a webpage
soup = BeautifulSoup(page.text, "html.parser")
links = []
for link in soup.find_all('a'):
    links.append(link.get('href'))

df = pd.DataFrame(recipe_urls)
df["recipe_urls"] = pd.DataFrame(recipe_urls)


# DataFrame to store the scraped URLs
df = pd.DataFrame({"recipe_urls":recipe_urls})
df["recipe_urls"] = "https://cuisine-facile.com" + df["recipe_urls"].astype("str")
# Appending 'df' to a main DataFrame 'init_urls_df'
recipe_url_df = recipe_url_df.append(df).copy()
df["recipe_urls"].to_csv(r"input/recipe_urls.csv", sep=',')

