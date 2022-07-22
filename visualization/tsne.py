import time
import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


if __name__ == '__main__':

	dataset = 'bloodmnist'
	# suffix = '_'.join(['pathological', 'vectors', 'unshuffled'])
	suffix = '_'.join(['vectors', 'unshuffled'])
	num_classes = 8
	subset = True and False

	crop = '24-0'
	encoder, D = 'resnet_small', 64
	norm = 'layer'
	grid_size = '5'
	pred_directions = '4'
	cpc_patch_aug = 'True'
	gray = '_colour'
	model_num = '200'

	df_path = f'../TrainedModels/{dataset}/trained_encoder_{encoder}_crop{crop}{gray}_grid{grid_size}_{norm}Norm_{pred_directions}dir_aug{cpc_patch_aug}_{model_num}{dataset}_{suffix}.csv'

	df = pd.read_csv(df_path)
	if subset:
		df = df.iloc[np.random.choice(len(df), 4000, replace=False)]
		# df = df[(df['label'] == 0) | (df['label'] == 3) | (df['label'] == 5)]
		# df = df[df['label'] > 0]

	# X = df[map(str, range(D))].values
	# X = (X - X.mean(axis=0)) / np.sqrt(X.var(axis=0))
	
	# for i in range(D):
	# 	df[str(i)] = X[:, i]

# '''
	perp = 1000

	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=500)
	tsne_results = tsne.fit_transform(df[map(str, range(D))].values)

	print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	df['tsne-2d-one'] = tsne_results[:,0]
	df['tsne-2d-two'] = tsne_results[:,1]

	plt.figure(figsize=(5,5))
	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="label",
		palette=sns.color_palette("bright", num_classes - (1 if 'pathological' in suffix else 0) - (0 if subset else 0)),
		data=df,
		# legend="full",
		alpha=0.75
	)
	plt.title(f'{dataset} ({encoder}) - Perplexity={perp}')
	plt.savefig(f'{dataset}-{encoder}-{perp}{"-a" if subset else ""}.png')
	# plt.show()
# '''
