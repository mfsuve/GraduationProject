from bs4 import BeautifulSoup
import requests
import urllib.request

page_url = 'https://www.goodreads.com/genres/most_read/science'
page_response = requests.get(page_url, timeout=5)
soup = BeautifulSoup(page_response.content, 'html.parser')

i = 0
for img in soup.findAll('img'):
	url = img.get('src')
	urllib.request.urlretrieve(url, '../newbooks/book_' + str(i) + '.jpg')
	i += 1
	print('book ' + str(i) + '\tis being fetched...')
	if i > 99:
		break
