from bs4 import BeautifulSoup
import requests
import urllib.request
import os
import cv2

folder = 'newbooks'
genre = 'engineering'
# Don't have to touch this
path = '../' + folder + '/' + genre + '/'

print('Connecting the URL...')
page_url = 'https://www.goodreads.com/genres/most_read/' + genre
try:
	page_response = requests.get(page_url, timeout=5)
except requests.packages.urllib3.exceptions.ConnectTimeoutError:
	print('Connection Error! Try again.')
	exit()
except requests.packages.urllib3.exceptions.MaxRetryError:
	print('Connection Error! Try again.')
	exit()
except requests.exceptions.ConnectTimeout:
	print('Connection Error! Try again.')
	exit()
soup = BeautifulSoup(page_response.content, 'html.parser')

print('Connected!')

# Check if the folder exist
if not os.path.isdir(path):
	print('Path does not exist, creating path...')
	os.mkdir(path)

	i = 0
	for img in soup.findAll('img'):
		url = img.get('src')
		book_name = 'book_' + str(i) + '.jpg'
		print(book_name + '\tis being fetched...')
		try:
			urllib.request.urlretrieve(url, path + book_name)
		except ValueError as e:
			print(e)
			continue
		i += 1
		if i > 89:
			break

	print('Converting all images to png')
	for name in os.listdir(path):
		print('Processing ' + name + '...')
		image = cv2.imread(path + name)
		cv2.imwrite(path + name[:-3] + 'png', image)
		os.remove(path + name)

	print('Done!')

else:
	print('Path exists, please delete folder named \'' + genre + '\' in order to continue.\nExitting...')