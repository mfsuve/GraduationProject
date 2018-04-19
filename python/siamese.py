import cv2
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# from scipy.misc import imread


data_path = '/home/mustafa/PCL_TUTORIAL/python/keras-oneshot/omniglot/python/'


def loadimgs(path,n=0):
	#if data not already unzipped, unzip it.
	if not os.path.exists(path):
		print("unzipping")
		os.chdir(data_path)
		os.system("unzip {}".format(path+".zip"))
	X = []
	y = []
	cat_dict = {}
	lang_dict = {}
	curr_y = n
	#we load every alphabet seperately so we can isolate them later
	for alphabet in os.listdir(path):
		print("loading alphabet: " + alphabet)
		lang_dict[alphabet] = [curr_y,None]
		alphabet_path = os.path.join(path,alphabet)
		#every letter/category has it's own column in the array, so  load seperately
		for letter in os.listdir(alphabet_path):
			cat_dict[curr_y] = (alphabet, letter)
			category_images=[]
			letter_path = os.path.join(alphabet_path, letter)
			for filename in os.listdir(letter_path):
				image_path = os.path.join(letter_path, filename)
				image = imread(image_path)
				category_images.append(image)
				y.append(curr_y)
			try:
				X.append(np.stack(category_images))
			#edge case  - last one
			except ValueError as e:
				print(e)
				print("error - category_images:", category_images)
			curr_y += 1
			lang_dict[alphabet][1] = curr_y - 1
	y = np.vstack(y)
	X = np.stack(X)
	return X, y, lang_dict




class Siamese_Loader:
	"""For loading batches and testing tasks to a siamese net"""
	def __init__(self, Xtrain, Xval):
		self.Xval = Xval
		self.Xtrain = Xtrain
		self.n_classes, self.n_examples, self.w, self.h = Xtrain.shape
		self.n_val, self.n_ex_val, _, _ = Xval.shape

	def get_batch(self, n):
		"""Create batch of n pairs, half same class, half different class"""
		categories = rng.choice(self.n_classes, size=(n, ), replace=False)
		pairs = [np.zeros((n, self.w, self.h, 1)) for i in range(2)]
		targets = np.zeros((n, ))
		targets[n//2:] = 1
		for i in range(n):
			category = categories[i]
			idx_1 = rng.randint(0, self.n_examples)
			pairs[0][i, :, :, :] = self.Xtrain[category, idx_1].reshape(self.w, self.h, 1)
			idx_2 = rng.randint(0, self.n_examples)

			category_2 = category if i >= n//2 else (category + rng.randint(1, self.n_classes)) % self.n_classes
			pairs[1][i, :, :, :] = self.Xtrain[category_2, idx_2].reshape(self.w, self.h, 1)
		return pairs, targets

	def make_oneshot_task(self, N):
		"""Create pairs of test image, support set for testing N way one-shot learning. """
		categories = rng.choice(self.n_val, size=(N, ), replace=False)
		indices = rng.randint(0, self.n_ex_val, size=(N, ))
		true_category = categories[0]
		ex1, ex2 = rng.choice(self.n_examples, replace=False, size=(2, ))
		test_image = np.asarray([self.Xval[true_category, ex1, :, :]]*N).reshape(N, self.w, self.h, 1)
		support_set = self.Xval[categories, indices, :, :]
		support_set[0, :, :] = self.Xval[true_category, ex2]
		support_set = support_set.reshape(N, self.w, self.h, 1)
		pairs = [test_image, support_set]
		targets = np.zeros((N, ))
		targets[0] = 1
		return pairs, targets

	def test_oneshot(self, model, N, k, verbose=0):
		"""Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
		pass
		n_correct = 0
		if verbose:
			print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k, N))
		for i in range(k):
			inputs, targets = self.make_oneshot_task(N)
			probs = model.predict(inputs)
			print('#### i:', i)
			if np.argmax(probs) == 0:
				n_correct += 1
		percent_correct = (100.0*n_correct / k)
		if verbose:
			print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N))
		return percent_correct


def W_init(shape, name=None):
	"""Initialize weights as in paper"""
	values = rng.normal(loc=0, scale=1e-2, size=shape)
	return K.variable(values, name=name)


def b_init(shape, name=None):
	"""Initialize bias as in paper"""
	values=rng.normal(loc=0.5, scale=1e-2, size=shape)
	return K.variable(values, name=name)


print('## Creating siamese net')
input_shape = (150, 100, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)

convnet = Sequential()
convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4), bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4), bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=W_init, bias_initializer=b_init))

encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

L1_distance = lambda x: K.abs(x[0]-x[1])
both = merge([encoded_l, encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input, right_input], output=prediction)

print('## Compiling siamese net')
optimizer = Adam(0.00006)
siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)
siamese_net.count_params()

# print('## images background loading')
# Xt, y, c = loadimgs('/home/mustafa/PCL_TUTORIAL/python/keras-oneshot/omniglot/python/images_background')
# print('Xt shape:', np.shape(Xt))
# print('y  shape:', np.shape(y))
# print('c  shape:', np.shape(c))
# print('## images evaluation loading')
# Xv, y, c = loadimgs('/home/mustafa/PCL_TUTORIAL/python/keras-oneshot/omniglot/python/images_evaluation')
# print('Xv shape:', np.shape(Xv))
# print('y  shape:', np.shape(y))
# print('c  shape:', np.shape(c))

def my_loader(path):
	i = 0
	Xt = []
	Xv = []
	for filename in os.listdir(path):
		filepath = os.path.join(path, filename)
		image = cv2.imread(filepath)
		image = cv2.resize(image, (100, 150), interpolation=cv2.INTER_CUBIC)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if i < 65:
			Xt.append(image)
		else:
			Xv.append(image)
		i += 1

	return np.expand_dims(np.stack(Xt), axis=1), np.expand_dims(np.stack(Xv), axis=1)

Xt, Xv = my_loader('/home/mustafa/PCL_TUTORIAL/python/finals/reduced/train')
print('Xt shape:', Xt.shape)
print('Xv shape:', Xv.shape)
loader = Siamese_Loader(Xt, Xv)

evaluate_every = 10
loss_every = 300
batch_size = 32
N_way = 20
n_val = 550
# siamese_net.load_weights("PATH")
best = 76.0
print('## Entering the loop')
for i in range(900000):
	(inputs, targets) = loader.get_batch(batch_size)


	# print('## i:', i)
	# for ii in range(len(targets)):
	# 	target = targets[ii]
	# 	plt.suptitle('Same!' if target == 1 else 'Different!')
	# 	plt.subplot(1, 2, 1)
	# 	print('input1 shape:', np.shape(inputs[0][ii, :, :, :]))
	# 	plt.imshow(np.reshape(inputs[0][ii, :, :, :], (150, 100)), cmap='gray')
	# 	plt.subplot(1, 2, 2)
	# 	plt.imshow(np.reshape(inputs[1][ii, :, :, :], (150, 100)), cmap='gray')
	# 	plt.show()

	loss = siamese_net.train_on_batch(inputs, targets)
	print('## i:', i, 'loss:', loss)
	# if i % evaluate_every == 0:
	# 	val_acc = loader.test_oneshot(siamese_net, N_way, n_val, verbose=True)
	# 	print('## val_acc:', val_acc)
	# 	if val_acc >= best:
	# 		print("saving")
	# 		# siamese_net.save('PATH')
	# 		best = val_acc
	#
	# if i % loss_every == 0:
	# 	print("iteration {}, training loss: {:.2f}, ".format(i, loss))