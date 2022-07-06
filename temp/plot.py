import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	for dataset in ['breastmnist', 'retinamnist', 'bloodmnist']:

		values = np.loadtxt(f'multi/{dataset}.txt')
		N = 20

		data = {}

		data['cpc', 'train', 'loss'] = values[0 : 2*N : 2, 0]
		data['cpc', 'train', 'accuracy'] = values[0 : 2*N : 2, 1]
		data['cpc', 'test', 'loss'] = values[1 : 2*N : 2, 0]
		data['cpc', 'test', 'accuracy'] = values[1 : 2*N : 2, 1]
		data['sup', 'train', 'loss'] = values[2*N : 4*N : 2, 0]
		data['sup', 'train', 'accuracy'] = values[2*N : 4*N : 2, 1]
		data['sup', 'test', 'loss'] = values[2*N + 1 : 4*N : 2, 0]
		data['sup', 'test', 'accuracy'] = values[2*N + 1 : 4*N : 2, 1]

		for score in ['loss', 'accuracy']:
			plt.figure(figsize=(9,9))
			legend = []

			for setting in ['cpc', 'sup']:
				for phase in ['train', 'test']:

					clip = (data[setting, phase, score][0]) if score == 'loss' else 100

					plt.plot(range(N), data[setting, phase, score].clip(max=clip))
					legend.append(f'{setting}_{phase}')

			plt.legend(legend)
			plt.xlabel('epochs')
			plt.ylabel(score)
			plt.title(f'{dataset}')
			plt.savefig(f'{dataset}-{score}')
