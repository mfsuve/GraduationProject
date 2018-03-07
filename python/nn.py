import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']
num_classes = len(names)


def load_data():
	path = 'finals/reduced/'
	train_img = []
	test_img = []
	for name in names:
		train_img.append(cv2.resize(cv2.imread(path + name + '.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
		test_img.append(cv2.resize(cv2.imread(path + name + '_test.png'), (100, 150), interpolation=cv2.INTER_CUBIC))

	return (np.array(train_img), np.arange(num_classes)), (np.array(test_img), np.arange(num_classes))


def create_model():
	m = Sequential()
	m.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(150, 100, 3)))
	m.add(Convolution2D(32, (3, 3), activation='relu'))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Dropout(0.25))

	m.add(Flatten())
	m.add(Dense(128, activation='relu'))
	m.add(Dropout(0.5))
	m.add(Dense(num_classes, activation='softmax'))

	return m


(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape(X_train.shape[0], 150, 100, 3)
X_test = X_test.reshape(X_test.shape[0], 150, 100, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# print(keras.backend.image_data_format())

model = create_model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=30, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)

# guesses = np.argmax(model.predict(X_test), axis=0)
wrong = False
for i in range(num_classes):
	guess = model.predict(X_test[i:i+1])
	# print('Guess is', names[np.argmax(guess)])
	print('Guess for %r is %r' % (names[y_test[i]], names[np.argmax(guess)]))
	if np.argmax(guess) == y_test[i]:
		print('\tCorrect!')
	else:
		print('\tIncorrect!')
		wrong = True

if not wrong:
	print('\nAll the guesses are correct!!!')
