#!/usr/bin/env python

import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
from threading import Thread

import time

global model
names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']

def load(model_name):
	global model
	while True:
		try:
			model = load_model(model_name)
			break
		except OSError:
			print('Model at ' + model_name + ' not found. Trying again in 30 seconds...')
			time.sleep(30)



def guess(file):
	global model
	image = cv2.imread(file)
	image = cv2.resize(image, (100, 150), interpolation=cv2.INTER_CUBIC)
	image = image.reshape(150, 100, 3)
	image = image.astype('float32') / 255
	predictions = model.predict(np.expand_dims(image, axis=0))
	fig = plt.figure()
	plt.title(names[np.argmax(predictions)])
	plt.imshow(image[:, :, ::-1])
	close = Thread(target=cls, args=(fig,))
	close.start()
	plt.show()
	close.join()



def cls(fig):
	time.sleep(5)
	print('Figure closed..')
	plt.close(fig)
