from matplotlib import pyplot as plt

def show_plot(history, title=None):
	if title is None:
		plt.suptitle('Tested on ' + file + ', Model Name: ' + model_name + '\nMode: ' + mode + ', Backend: ' + backend)
	else:
		plt.suptitle(title)

	try:
		plt.subplot(221)
		plt.ylim(0, 1.05)
		plt.title('acc')
		plt.plot(history['acc'])

		plt.subplot(222)
		plt.ylim(0, 1.05)
		plt.title('val_acc')
		plt.plot(history['val_acc'])

		plt.subplot(223)
		plt.ylim(0, 10)
		plt.title('loss')
		plt.plot(history['loss'])

		plt.subplot(224)
		plt.ylim(0, 10)
		plt.title('val_loss')
		plt.plot(history['val_loss'])
	except KeyError:
		plt.subplot(121)
		plt.ylim(0, 1.05)
		plt.title('acc')
		plt.plot(history['acc'])

		plt.subplot(122)
		plt.ylim(0, 10)
		plt.title('loss')
		plt.plot(history['loss'])

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()