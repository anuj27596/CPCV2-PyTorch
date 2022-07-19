import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	

	directory = 'retinamnist-pipelined'

	datasets = ['breastmnist', 'retinamnist', 'bloodmnist'][1:2]
	settings = ['cpc', 'sup'][1:]
	suffix = '-'.join(['HvP'])

	N = 50
	seeds = 1

	for dataset in datasets:

		data = {}

		for setting in settings:
			values = np.loadtxt(os.path.join(directory, f'{dataset}-{setting}-{suffix}.txt'))

			data[setting, 'train', 'loss'] = values[0::2, 0].reshape((seeds, N)).mean(axis=0)
			data[setting, 'train', 'accuracy'] = values[0::2, 1].reshape((seeds, N)).mean(axis=0)
			data[setting, 'test', 'loss'] = values[1::2, 0].reshape((seeds, N)).mean(axis=0)
			data[setting, 'test', 'accuracy'] = values[1::2, 1].reshape((seeds, N)).mean(axis=0)

		for score in ['loss', 'accuracy']:
			plt.figure(figsize=(5,5))
			legend = []

			for setting in settings:
				for phase in ['train', 'test']:
					# clip = (data[setting, phase, score][0]) if score == 'loss' else 100
					clip = 4 if score == 'loss' else 100

					plt.plot(range(N), data[setting, phase, score].clip(max=clip))
					legend.append(f'{setting}_{phase}')

			plt.legend(legend)
			plt.xlabel('epochs')
			plt.ylabel(score)
			plt.title(f'{dataset} - {suffix}')
			plt.savefig(os.path.join(directory, f'{dataset}-{score}-{suffix}.png'))
			# plt.show()
