import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

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
	m.add(Dense(num_classes, activation='softmax'))

	return m

def examine(guess=None):
	if guess is not None:
		plt.imshow(X_test[guess, ::-1, :, ::-1])
		plt.show()
		predictions = model.predict(X_test_3ch[guess:guess+1])
		print('Guess is', names[np.argmax(predictions)])
	else:
		guesses = np.argmax(model.predict(X_test_3ch), axis=1)
		wrong = False
		for i in range(len(guesses)):
			plt.title('Guess is %r' % (names[guesses[i]]))
			plt.imshow(X_test[i, :, ::-1, ::-1])
			plt.show()
			if guesses[i] == y_test[i]:
				print('\tCorrect!')
			else:
				print('\tIncorrect!')
				wrong = True

		if not wrong:
			print('\nAll the guesses are correct!!!')

		# for guess in range(num_classes):
		# 	plt.imshow(X_test[guess, :, ::-1, ::-1])
		# 	plt.show()
		# 	predictions = model.predict(X_test_3ch[guess:guess+1])
		# 	print('Guess is', names[np.argmax(predictions)])

	scores = model.evaluate(X_test_3ch, Y_test)
	print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

	plt.plot(history.history['val_acc'])
	plt.show()


# Load images
(X_train, y_train), (X_test, y_test) = load_data()
# Adjust sizes
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

X_train_3ch = X_train.reshape(num_classes, 150, 100, 3)
X_test_3ch = X_test.reshape(num_classes, 150, 100, 3)

X_train_3ch = X_train_3ch.astype('float32') / 255
X_test_3ch = X_test_3ch.astype('float32') / 255

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_datagen.fit(X_train_3ch)
train_generator = train_datagen.flow(X_train_3ch, Y_train, batch_size=32)

model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit_generator(train_generator, steps_per_epoch=20, epochs=30, validation_data=(X_test_3ch, Y_test))
examine(guess=None)
