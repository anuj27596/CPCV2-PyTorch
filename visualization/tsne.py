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

	dataset = 'retinamnist'
	num_classes = 5
	epochs = '20'
	crop = '24-0'
	encoder = 'resnet18'
	norm = 'layer'
	grid_size = '5'
	pred_directions = '4'
	cpc_patch_aug = 'True'
	gray = '_colour'
	model_num = '10'

	df_path = f'../TrainedModels/{dataset}/trained_encoder_{encoder}_crop{crop}{gray}_grid{grid_size}_{norm}Norm_{pred_directions}dir_aug{cpc_patch_aug}_{model_num}{dataset}_vectors.csv'

	df = pd.read_csv(df_path)
	# df = df[df['label'] > 1]

	# X = df[map(str, range(512))].values
	# X = (X - X.mean(axis=0)) / np.sqrt(X.var(axis=0))
	
	# for i in range(512):
	# 	df[str(i)] = X[:, i]


	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=400, n_iter=1000)
	tsne_results = tsne.fit_transform(df[map(str, range(512))].values)

	print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	df['tsne-2d-one'] = tsne_results[:,0]
	df['tsne-2d-two'] = tsne_results[:,1]

	plt.figure(figsize=(16,9))
	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="label",
		palette=sns.color_palette("hls", num_classes),
		data=df,
		legend="full",
		alpha=0.8
	)
	plt.show()
