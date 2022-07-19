import time
import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


if __name__ == '__main__':

	dataset = 'bloodmnist'
	num_classes = 8
	subset = False

	crop = '24-0'
	encoder = 'resnet18'
	norm = 'layer'
	grid_size = '5'
	pred_directions = '4'
	cpc_patch_aug = 'True'
	gray = ''#_colour'
	model_num = '100'

	df_path = f'../TrainedModels/{dataset}/trained_encoder_{encoder}_crop{crop}{gray}_grid{grid_size}_{norm}Norm_{pred_directions}dir_aug{cpc_patch_aug}_{model_num}{dataset}_vectors_unshuffled.csv'
	# df_path = 'bloodmnist-gray-unshuffled-tsne-clusters.csv'

	df = pd.read_csv(df_path)


	# # TSNE

	if subset:
		df = df.iloc[np.random.choice(len(df), 4000, replace=False)]
		# df = df[(df['label'] == 0) | (df['label'] == 3) | (df['label'] == 5)]
		# df = df[df['label'] > 0]

	# X = df[map(str, range(512))].values
	# X = (X - X.mean(axis=0)) / np.sqrt(X.var(axis=0))
	
	# for i in range(512):
	# 	df[str(i)] = X[:, i]

	perp = 1000

	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=500)
	tsne_results = tsne.fit_transform(df[map(str, range(512))].values)

	print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	df['tsne-2d-one'] = tsne_results[:,0]
	df['tsne-2d-two'] = tsne_results[:,1]

	# # df.to_csv('bloodmnist-tsne.csv', index=False)


	K = 16

	# # KMeans

	X = df[['tsne-2d-one', 'tsne-2d-two']].values

	kmeans = KMeans(n_clusters=K).fit(X)

	# palette = ['blue', 'green', 'red', 'yellow']
	# colors = [palette[i] for i in kmeans.labels_]

	df['cluster'] = kmeans.labels_


	# # KMeans (High-D)

	X = df[map(str, range(512))].values

	# cluster_ids = df['cluster']
	# centroids = np.array([X[cluster_ids == i].mean(axis=0) for i in range(K)])
	# kmeans = KMeans(n_clusters=K, init=centroids).fit(X)

	kmeans = KMeans(n_clusters=K).fit(X)

	df['hd-cluster'] = kmeans.labels_


	df.to_csv('bloodmnist-gray-unshuffled-tsne-clusters.csv', index=False)

	# # Plot

	plt.figure(figsize=(5,5))

	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="cluster", palette=sns.color_palette("bright", K),
		data=df,
		alpha=0.75
	)

	# plt.scatter(X[:, 0], X[:, 1], c=colors)

	plt.title(f'{dataset} - clusters')
	# plt.savefig(f'{dataset}{"-a" if subset else ""}.png')
	plt.show()


	plt.figure(figsize=(5,5))

	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="hd-cluster", palette=sns.color_palette("bright", K),
		data=df,
		alpha=0.75
	)

	# plt.scatter(X[:, 0], X[:, 1], c=colors)

	plt.title(f'{dataset} - hd-clusters')
	# plt.savefig(f'{dataset}{"-a" if subset else ""}.png')
	plt.show()


	plt.figure(figsize=(5,5))

	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="label", palette=sns.color_palette("bright", num_classes -1 - (5 if subset else 0)),
		data=df,
		alpha=0.75
	)

	# plt.scatter(X[:, 0], X[:, 1], c=colors)

	plt.title(f'{dataset} - labels')
	# plt.savefig(f'{dataset}{"-a" if subset else ""}.png')
	plt.show()

