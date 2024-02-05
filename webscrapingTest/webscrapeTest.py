# import requests

# URL = "https://realpython.github.io/fake-jobs/"
# page = requests.get(URL)

# print(page.text)

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://hoopshype.com/salaries/players/"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

results = soup.find(id="content-container")

names = results.find_all("td", class_="name")

arr = []
for name in names:
    arr.append(name.text)

print(arr)
arr = pd.DataFrame(arr)
print(arr.info())
print(arr.head())