import cv2
import os

path = '../newbooks/'

for name in os.listdir(path):
	image = cv2.imread(path + name)
	cv2.imwrite(path + name[:-3] + 'png', image)
	os.remove(path + name)
