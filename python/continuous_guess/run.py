#!/usr/bin/env python

import time
import signal
import os
import guess

exit = False


def signal_handler(signal, frame):
	global exit
	exit = True


signal.signal(signal.SIGINT, signal_handler)
guess.load('../saved_weights/normal_tensorflow_vgg16_10x5.h5')

print('Model is loaded.')

while not exit:
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
