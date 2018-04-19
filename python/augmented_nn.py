import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, ThresholdedReLU, BatchNormalization, merge
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']
num_test_classes = len(names)
num_train_classes = 0
global mode, model, X_train_3ch, X_test_3ch, Y_train, Y_test


def load_data():
	train_path = 'finals/reduced/train/'
	test_path = 'finals/reduced/test/'
	train_img = []
	test_img = []
	num_scrapped_book = 100
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
	top_model.add(ZeroPadding2D((1, 1), input_shape=base_model.output_shape[1:]))
	top_model.add(Conv2D(32, (3, 3), activation='relu'))
	top_model.add(BatchNormalization())
	top_model.add(ThresholdedReLU(0))

	top_model.add(ZeroPadding2D((1, 1)))
	top_model.add(Conv2D(32, (3, 3), activation='relu'))
	top_model.add(BatchNormalization())
	top_model.add(ThresholdedReLU(0))

	top_model.add(ZeroPadding2D((1, 1)))
	top_model.add(Conv2D(32, (3, 3), activation='relu'))
	top_model.add(BatchNormalization())
	top_model.add(ThresholdedReLU(0))

	top_model.add(MaxPooling2D(pool_size=(2, 2)))

	top_model.add(Flatten())
	top_model.add(BatchNormalization())
	top_model.add(Dense(num_train_classes, activation='softmax'))

	m = Model(inputs=base_model.input, outputs=top_model(base_model.output))

	# m.summary()

	return m


def siamese_generator(X, datagen, batch_size=32):
	cls_num = X.shape[0]
	batch_size = min(batch_size, cls_num - 1)
	categories = np.random.choice(cls_num, size=(batch_size,), replace=False)
	pairs = [np.zeros((batch_size, 150, 100, 3)) for i in range(2)]
	targets = np.zeros((batch_size,))
	targets[batch_size//2:] = 1

	while True:
		for i in range(batch_size):
			category = categories[i]
			pairs[0][i, :, :, :] = datagen.random_transform(X[category])
			category_2 = category if i >= batch_size // 2 else (category + np.random.randint(1, cls_num)) % cls_num
			pairs[1][i, :, :, :] = datagen.random_transform(X[category_2])
		yield (pairs, targets)


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
	# losses = []

	# for i in range(100):
	# 	(inputs, targets) = siamese_batch(X_train_3ch, train_datagen, 9)
	#
	# 	for g in range(9):
	# 		plt.suptitle('Same' if targets[g] == 1 else 'Different')
	# 		plt.subplot(1, 2, 1)
	# 		plt.imshow(inputs[0][g, :, :, ::-1])
	#
	# 		plt.subplot(1, 2, 2)
	# 		plt.imshow(inputs[1][g, :, :, ::-1])
	# 		plt.show()
	#
	# 	loss = model.train_on_batch(inputs, targets)
	#
	# 	### TODO: Calculate Accuracy Here !!!
	#
	# 	losses.append(loss)
	# 	print('iteration {},\ttraining loss: {}'.format(i, loss))
	#
	# return {'loss': losses, 'acc': [], 'val_loss': [], 'val_acc': []}

	train_generator = siamese_generator(X_train_3ch, train_datagen)
	# return model.fit_generator(train_generator, steps_per_epoch=30, epochs=200, validation_data=(X_test_3ch, Y_test))

	# Testing the generator
	# for (pairs, targets) in train_generator:
	# 	for i in range(len(targets)):
	# 		target = targets[i]
	# 		plt.suptitle('Same!' if target == 1 else 'Different!')
	# 		plt.subplot(1, 2, 1)
	# 		plt.imshow(pairs[0][i, :, :, ::-1])
	# 		plt.subplot(1, 2, 2)
	# 		plt.imshow(pairs[1][i, :, :, ::-1])
	# 		plt.show()

	# i = 0
	# for (pairs, targets) in train_generator:
	# 	print('inputs shape:', np.shape(pairs))
	# 	print('target shape:', np.shape(targets))
	# 	print('inputs type:', type(pairs))
	# 	print('target type:', type(targets))
	#
	# 	loss = model.train_on_batch(pairs, targets)
	# 	print('## i:', i, 'loss:', loss)
	# 	i += 1

	# fit_generator asks for tuples I think, look to see what type train_generator sends
	return model.fit_generator(train_generator, steps_per_epoch=2, epochs=10)
	# return model.fit_generator(train_generator, steps_per_epoch=30, epochs=200)


def normal_fit():
	global mode
	mode = 'normal'
	return model.fit(X_train_3ch, Y_train, batch_size=32, epochs=600, validation_data=(X_test_3ch, Y_test))


def siamese(smodel):
	input_shape = (150, 100, 3)
	left_input = Input(input_shape)
	right_input = Input(input_shape)

	encoded_l = smodel(left_input)
	encoded_r = smodel(right_input)

	L1 = lambda x: K.abs(x[0] - x[1])
	both = merge([encoded_l, encoded_r], mode=L1, output_shape=lambda x: x[0])

	prediction = Dense(1, activation='softmax')(both)

	return Model(inputs=[left_input, right_input], outputs=prediction)


def run(lr=0.001, augmented=True, modelno=3):  # If modelno changes, change the model_name (vgg16 part)
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

	model = siamese(model)

	sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	# Because of Siamese, I use the one above
	# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	# K.set_value(model.optimizer.lr, lr)  # it was 0.01

	model.count_params()

	print('learning rate is', K.get_value(model.optimizer.lr))

	if augmented:
		history = augmentation_fit()
	else:
		history = normal_fit()

	model_name = 'siamese_vgg16_' + mode + '_' + K.backend()
	# model_name = '195x10_vgg16_' + mode + '_' + K.backend() + '_lr_' + str(lr) + '_longer'
	pickle.dump(history.history, open('siamese_histories/' + model_name + '.p', 'wb'))

	# model.save('saved_weights/' + model_name + '.h5')


# for i in np.arange(0.001, 0.011, 0.001):
# 	lr = np.floor(i * 1000) / 1000.0
# 	run(lr)

run(lr=0.003)
