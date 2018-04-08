#!/usr/bin/env python

import time
import signal
import os
from guess import Guess as guess

Exit = False
guess.autoclose = False


def signal_handler(signal, frame):
	global Exit
	Exit = True


signal.signal(signal.SIGINT, signal_handler)
guess.load('../saved_weights/augmentation_tensorflow_vgg16_30x10_lrdropped_longer.h5')

print('Model is loaded.')

while not Exit:
	files = os.listdir('image')
	for file in files:
		print('Processing on ' + file)
		guess.guess('image/' + file)
		try:
			os.remove('image/' + file)
		except OSError:
			pass
	print('Nothing to guess')
	time.sleep(2)
