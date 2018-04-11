import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, ThresholdedReLU, BatchNormalization
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']
num_test_classes = len(names)
num_train_classes = 0
global mode, model, X_train_3ch, X_test_3ch, Y_train, Y_test


def load_data():
	train_path = 'finals/reduced/train/'
	test_path = 'finals/reduced/test/'
	train_img = []
	test_img = []
	num_scrapped_book = 15
	# Collect initials
	for name in names:
#		train_img.append(cv2.resize(cv2.imread(train_path + name + '1.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
		train_img.append(cv2.resize(cv2.imread(train_path + name + '2.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
		test_img.append(cv2.resize(cv2.imread(test_path + name + '_test1.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
		test_img.append(cv2.resize(cv2.imread(test_path + name + '_test2.png'), (100, 150), interpolation=cv2.INTER_CUBIC))

	first_books_test_labels = [val for val in range(num_test_classes) for _ in range(2)]

	# Collect web-scrapped books
	for i in range(num_scrapped_book):
		train_img.append(cv2.resize(cv2.imread(train_path + 'book_' + str(i) + '.png'), (100, 150), interpolation=cv2.INTER_CUBIC))

	global num_train_classes
	num_train_classes = num_test_classes + num_scrapped_book

	return (np.array(train_img), np.arange(num_train_classes)), (np.array(test_img), np.array(first_books_test_labels))


def create_model1():
	m = Sequential()

	m.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100, 3)))
	m.add(Conv2D(32, (3, 3), activation='relu'))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Dropout(0.25))

	m.add(Conv2D(32, (3, 3), activation='relu'))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Dropout(0.25))

	m.add(Flatten())
	m.add(Dense(128, activation='relu'))
	m.add(Dropout(0.5))
	m.add(Dense(num_train_classes, activation='softmax'))

	return m


def create_model2():
	m = Sequential()

	# (Conv -> Relu -> Conv -> Relu -> MaxPool) * 3 -> Flat -> Dense

	for u in range(3):
		if u == 0:
			m.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100, 3)))
		else:
			m.add(Conv2D(32, (3, 3), activation='relu'))
		m.add(ThresholdedReLU(0))
		m.add(Conv2D(32, (3, 3), activation='relu'))
		m.add(ThresholdedReLU(0))

		m.add(MaxPooling2D(pool_size=(2, 2)))

	m.add(Flatten())
	m.add(Dense(num_train_classes, activation='softmax'))

	return m


def create_model_vgg16():
	from keras.applications.vgg16 import VGG16

	input_tensor = Input(shape=(150, 100, 3))
	base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

	for layer in base_model.layers[:15]:
		layer.trainable = False

	top_model = Sequential()
	top_model.add(ZeroPadding2D((4, 4), input_shape=base_model.output_shape[1:]))
	top_model.add(Conv2D(32, (9, 9), activation='relu'))
	top_model.add(MaxPooling2D(pool_size=(4, 4)))
	top_model.add(BatchNormalization())
	top_model.add(ThresholdedReLU(0))

	top_model.add(ZeroPadding2D((4, 4)))
	top_model.add(Conv2D(32, (9, 9), activation='relu'))
	top_model.add(MaxPooling2D(pool_size=(4, 4)))
	top_model.add(BatchNormalization())
	top_model.add(ThresholdedReLU(0))

	top_model.add(ZeroPadding2D((4, 4)))
	top_model.add(Conv2D(32, (9, 9), activation='relu'))
	top_model.add(MaxPooling2D(pool_size=(4, 4)))
	top_model.add(BatchNormalization())
	top_model.add(ThresholdedReLU(0))

	top_model.add(Flatten())
	top_model.add(BatchNormalization())
	top_model.add(Dense(num_train_classes, activation='softmax'))

	m = Model(inputs=base_model.input, outputs=top_model(base_model.output))

	return m


def augmentation_fit():
	global mode
	mode = 'augmentation'
	train_datagen = ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='constant')  # Constant zero

	train_datagen.fit(X_train_3ch)
	train_generator = train_datagen.flow(X_train_3ch, Y_train, batch_size=32)

	return model.fit_generator(train_generator, steps_per_epoch=30, epochs=200, validation_data=(X_test_3ch, Y_test))


def normal_fit():
	global mode
	mode = 'normal'
	return model.fit(X_train_3ch, Y_train, batch_size=32, epochs=600, validation_data=(X_test_3ch, Y_test))


def run(lr=0.001, augmented=True, modelno=3):  # If modelno changes, change the model_name
	global model, X_train_3ch, X_test_3ch, Y_train, Y_test
	# Load images
	(X_train, y_train), (X_test, y_test) = load_data()

	# Adjust sizes
	Y_train = np_utils.to_categorical(y_train, num_train_classes)
	Y_test = np_utils.to_categorical(y_test, num_train_classes)

	X_train_3ch = X_train.reshape(X_train.shape[0], 150, 100, 3)
	X_test_3ch = X_test.reshape(X_test.shape[0], 150, 100, 3)

	X_train_3ch = X_train_3ch.astype('float32') / 255
	X_test_3ch = X_test_3ch.astype('float32') / 255

	if modelno == 1:
		model = create_model1()
	elif modelno == 2:
		model = create_model2()
	else:  # default
		model = create_model_vgg16()
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	K.set_value(model.optimizer.lr, lr)  # it was 0.01

	if augmented:
		history = augmentation_fit()
	else:
		history = normal_fit()

	model_name = '195x10_vgg16_' + mode + '_' + K.backend() + '_lr_' + str(lr) + '_longer'
	pickle.dump(history.history, open('histories/' + model_name + '.p', 'wb'))

	# model.save('saved_weights/' + model_name + '.h5')


# for i in np.arange(0.001, 0.011, 0.001):
# 	lr = np.floor(i * 1000) / 1000.0
# 	run(lr)

run(lr=0.003)
