from show import show_plot
import pickle
import os

inFormat = False
folder = 'sc_histories/'

augmented = True
lrdropped = True
file = '15x10'
model = 3
backend = 'tensorflow'
# backend = 'theano'

mode = 'augmentation' if augmented else 'normal'
lr = 'lrdropped_' if lrdropped else ''
model_name = ['model1', 'model2', 'vgg16'][model - 1]


if inFormat:
	for i in range(1, 6):
		try:
			history = pickle.load(open('../' + file +'histories/' + model_name + '/' + backend + '/' + mode + '_' + backend + '_' + lr + str(i) + '.p', 'rb'))
		except FileNotFoundError:
			break
		show_plot(history)

else:
	dirs = os.listdir('../' + folder)
	dirs.sort()
	for file_name in dirs:
		history = pickle.load(open('../' + folder + file_name, 'rb'))
		t = file_name[:-2].split('_')
		print(t)
		title = 'Tested on {0} {1}, Mode: {2}\nBackend: {3}, Learning Rate: {4}'.format(t[0], t[1], t[2], t[3], t[5])
		title += ("\n" + " ".join(t[6:])) if len(t) > 6 else ""
		show_plot(history, title=title)
