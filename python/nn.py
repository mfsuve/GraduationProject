import cv2
from matplotlib import pyplot as plt
from keras.models import load_model, Model
import numpy as np
import os

names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']

layer_name = 'sequential_1'
model = load_model("saved_weights/augmentation_tensorflow_vgg16_110x5_lrdropped.h5")

model.get_layer('sequential_1').summary()
# model.summary()

inter = Model(inputs=model.input, outputs=model.get_layer(layer_name).get_output_at(5))


def guess(file, ii=0):
	image = cv2.imread(file)
	image = cv2.resize(image, (100, 150), interpolation=cv2.INTER_CUBIC)
	image = image.reshape(150, 100, 3)
	image = image.astype('float32') / 255
	predictions = inter.predict(np.expand_dims(image, axis=0))
	print('#################### size:', predictions.shape)
	index = np.argmax(predictions)
	if isinstance(index, tuple):
		index = index[0]
	try:
		plt.title(names[index])
	except IndexError:
		plt.title('book_' + str(index - 5))
	for i in range(0, predictions.shape[3] - 3, 4):
		img = predictions[0, :, :, i:i+3] * 256
		cv2.imwrite('layers_visualize/' + layer_name + '/img' + str(int(i / 4 + ii * 1000)) + '.png', img)
		# plt.imshow(img)
		# plt.show()

try:
	os.mkdir('layers_visualize/' + layer_name)
except FileExistsError:
	pass
for (name, index) in zip(names, range(len(names))):
	guess('/home/mustafa/PCL_TUTORIAL/python/finals/reduced/test/' + name + '_test1.png', index)
