from bs4 import BeautifulSoup
import requests

website = 'https://atlas.emory.edu'

results = requests.get(website)

content = results.text

soup = BeautifulSoup(content, 'lxml')
# print(soup.prettify()) # Print site contents

# ID, class name, tag name (CSS selector), x path

box = soup.find('div', {'id': 'crit-content-1464624409188', 'class': 'section__content'})

formGroup = box.find('div', {'class': 'form-group'})
print(formGroup)

