#!/usr/bin/env python

import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
from threading import Thread

import time

names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']

class Guess:
	autoclose = True
	model = None

	@classmethod
	def load(cls, model_name):
		global model
		while True:
			try:
				cls.model = load_model(model_name)
				break
			except OSError:
				print('Model at ' + model_name + ' not found. Trying again in 30 seconds...')
				time.sleep(30)

	@classmethod
	def guess(cls, file):
		image = cv2.imread(file)
		image = cv2.resize(image, (100, 150), interpolation=cv2.INTER_CUBIC)
		image = image.reshape(150, 100, 3)
		image = image.astype('float32') / 255
		predictions = cls.model.predict(np.expand_dims(image, axis=0))
		fig = plt.figure()
		index = np.argmax(predictions)
		if isinstance(index, tuple):
			index = index[0]
		try:
			plt.title(names[index])
		except IndexError:
			plt.title('book_' + str(index - 5))
		plt.imshow(image[:, :, ::-1])
		closeThread = Thread(target=cls.close, args=(fig,))
		closeThread.start()
		plt.show()
		closeThread.join()

	@classmethod
	def close(cls, fig):
		if cls.autoclose:
			time.sleep(5)
			print('Figure closed..')
			plt.close(fig)
