from bs4 import BeautifulSoup
import requests

website = 'https://subslikescript.com/movie/TItanic-120338'
result = requests.get(website)
content = result.text

soup = BeautifulSoup(content, 'lxml')
# print(soup.prettify()) # Print site contents

# ID, class name, tag name (CSS selector), x path

box = soup.find('article', class_= 'main-article')

title = box.find('h1').get_text()

# print(title)

transcript = box.find('div', class_='full-script').get_text(strip=True, separator=' ')
# print(transcript)

with open(f'{title}.txt', 'w') as file:  # title + '.txt'
    file.write(transcript)