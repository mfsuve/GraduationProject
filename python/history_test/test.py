from matplotlib import pyplot as plt
import pickle
import os

inFormat = True
folder = 'lr_test_histories/'

augmented = True
lrdropped = True
file = '15x10'
model = 3
backend = 'tensorflow'
# backend = 'theano'

mode = 'augmentation' if augmented else 'normal'
lr = 'lrdropped_' if lrdropped else ''
model_name = ['model1', 'model2', 'vgg16'][model - 1]

def show_plot(history, title=None):
	if title is None:
		plt.suptitle('Tested on ' + file + ', Model Name: ' + model_name + '\nMode: ' + mode + ', Backend: ' + backend)
	else:
		plt.suptitle(title)

	plt.subplot(221)
	plt.ylim(0, 1.05)
	plt.title('acc')
	plt.plot(history['acc'])

	plt.subplot(222)
	plt.ylim(0, 1.05)
	plt.title('val_acc')
	plt.plot(history['val_acc'])

	# ylim = max(history['loss'] + history['val_loss']) + 0.05

	plt.subplot(223)
	plt.ylim(0, 10)
	plt.title('loss')
	plt.plot(history['loss'])

	plt.subplot(224)
	plt.ylim(0, 10)
	plt.title('val_loss')
	plt.plot(history['val_loss'])

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

if inFormat:
	for i in range(1, 6):
		try:
			history = pickle.load(open('../'+ file +'histories/' + model_name + '/' + backend + '/' + mode +  '_' + backend + '_' + lr + str(i) + '.p', 'rb'))
		except FileNotFoundError:
			break
		show_plot(history)

else:
	for file_name in os.listdir('../' + folder):
		history = pickle.load(open('../' + folder + file_name, 'rb'))
		t = file_name[:-2].split('_')
		title = 'Tested on {0} {1}, Mode: {2}\nBackend: {3}, Learning Rate: {4}'.format(t[0], t[1], t[2], t[3], t[5])
		show_plot(history, title=title)
