import cv2
import numpy as np
from matplotlib.pyplot import imshow, show, title

names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']
test = True

def autocrop(image, mask=None, threshold=0):
	if mask is None:
		mask = image

	if len(mask.shape) == 3:
		flatimage = np.max(mask, 2)
	else:
		flatimage = mask
	assert len(flatimage.shape) == 2

	rows = np.where(np.max(flatimage, 0) > threshold)[0]
	if rows.size:
		cols = np.where(np.max(flatimage, 1) > threshold)[0]
		image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
	else:
		image = image[:1, :1]

	return image


def autorotate(image, mask=None, threshold=0, isgray=True):
	h = np.size(image, axis=0)
	w = np.size(image, axis=1)

	# wasNone = mask is None
	#
	# if wasNone:
	# 	mask = image
	#
	# print('mask:', wasNone)

	if isgray:
		grayimage = mask
		print('mask')
	else:
		grayimage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		print('gray')

	# print(grayimage.shape)
	# print('h:', h, 'w:', w)

	sil = []
	for ii in range(h):
		for jj in range(w):
			if grayimage[ii][jj] > threshold:
				sil.append((ii, jj))

	sil = np.array(sil, dtype=np.float64)

	if len(sil) == 0:
		return image, mask

	mean = np.mean(sil, axis=0)
	sil[:, 0] -= mean[0]
	sil[:, 1] -= mean[1]

	sil = np.transpose(sil)

	cov = np.cov(sil)
	x, e, v = np.linalg.svd(cov)
	maxv = v[:, np.argmax(e)]
	# to bring the vector into the upper half of the coordinate plane
	if maxv[1] < 0:
		maxv *= -1

	angle = np.arctan2(maxv[1], maxv[0]) * 180 / np.pi

	if angle > 90:
		angle -= 180

	(cX, cY) = (int(w/2), int(h/2))

	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	newimage = cv2.warpAffine(image, M, (nW, nH))
	mask = cv2.warpAffine(mask, M, (nW, nH))

	# if wasNone:
	# 	return newimage
	# else:
	# 	return newimage, mask
	return newimage, mask

def largest_component(image):
	image = image.astype('uint8')
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
	sizes = stats[:, -1]

	if nb_components <= 1:
		return image

	max_label = 1
	max_size = sizes[1]
	for i in range(2, nb_components):
		if sizes[i] > max_size:
			max_label = i
			max_size = sizes[i]
	
	img2 = np.zeros(image.shape)
	img2 = img2.astype('uint8')
	img2[output == max_label] = 255
	return img2


def draw(contours, image):
	img2 = np.zeros(image.shape)
	img2 = img2.astype('uint8')

	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(img2, [box], 0, (255, 255, 255), -1)

	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	return img2


def display(img):
	imshow(img, cmap='gray')
	show()


def findBookEdges(image):
	image = image.astype('uint8')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
	opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((30, 30), np.uint8))
	opened = largest_component(opened)
	blurred = cv2.GaussianBlur(opened, (15, 15), 0)
	# blurred = autorotate(blurred, isgray=True)
	edges = cv2.Canny(blurred, 0, 1, apertureSize=3)
	# blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
	# image[edges>0] = [255,0,0]

	# contours, hierarchy = cv2.findContours(edges, 1, 2)
	return edges


# if not test:
# 	names = [(name + x) for name in names for x in ['1', '2']]
#
# for name in names:
# 	if test:
# 		name = name + '_test'
#
# 	print(name + ' processing...')




import os
import time

while True:
	for name in os.listdir('finals/temp'):

		print('preprocessing ' + name + ' ...')

		image = cv2.imread('finals/temp/' + name)
		# image = cv2.imread('../build/readpcd/png_files/' + name + '.png')
		edges = findBookEdges(image)

		im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		mask = draw(contours, image)

		image, mask = autorotate(image, mask)
		image = autocrop(image, mask)

		# epsilon = 0.1*cv2.arcLength(cnt,True)
		# approx = cv2.approxPolyDP(cnt,epsilon,True)

		# cv2.drawContours(image, approx, -1, (0,255,0), 3)

		# x,y,w,h = cv2.boundingRect(cnt)
		# cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

		# print(M)

		#title(name)
		#imshow(image[:,:,::-1])
		#imshow(blurred, cmap='gray')
		#imshow(mask, cmap='gray')
		#show()

		#dst = cv2.cornerHarris(edges,25,25,0.04)
		# result is dilated for marking the corners, not important
		#dst = cv2.dilate(dst,None)

		# Threshold for an optimal value, it may vary depending on the image.
		#image[dst>0.01*dst.max()]=[0,0,255]

		# lines = cv2.HoughLines(edges,1,np.pi/180,30)
		# for line in lines[:1]:
		# 	for rho,theta in line:
		# 		a = np.cos(theta)
		# 		b = np.sin(theta)
		# 		x0 = a*rho
		# 		y0 = b*rho
		# 		x1 = int(x0 + 1000*(-b))
		# 		y1 = int(y0 + 1000*(a))
		# 		x2 = int(x0 - 1000*(-b))
		# 		y2 = int(y0 - 1000*(a))

				#cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
		print('Done!')

		image = cv2.resize(image, (100, 150), interpolation=cv2.INTER_CUBIC)
		# cv2.imwrite('continuous_guess/image/' + name + '.png', image)
		cv2.imwrite('continuous_guess/image/' + name, image)
		time.sleep(2)
		os.remove('finals/temp/' + name)

	print('Nothing found..')
	time.sleep(3)
