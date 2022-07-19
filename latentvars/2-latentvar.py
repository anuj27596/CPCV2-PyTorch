import time
import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
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

	# df_path = f'../TrainedModels/{dataset}/trained_encoder_{encoder}_crop{crop}{gray}_grid{grid_size}_{norm}Norm_{pred_directions}dir_aug{cpc_patch_aug}_{model_num}{dataset}_vectors_unshuffled.csv'
	
	# df_path = 'bloodmnist-gray-all.csv'
	
	df_path = 'bloodmnist-all-unshuffled-tsne-clusters.csv'
	# df_path = 'bloodmnist-gray-all-unshuffled-tsne-clusters.csv'

	
	df = pd.read_csv(df_path)

	
	# df_colour = pd.read_csv('bloodmnist-all-unshuffled-tsne-clusters.csv')
	# df['colour-cluster'] = df_colour['hd-cluster'][(df_colour['label'] > 0) | (df_colour['phase'] == 'test')]

	
	# df = df[df['phase'] == 'train']
	# num_classes -= 1

	# phase = ['train'] * 11107 + ['test'] * 3421
	# df['phase'] = phase


	# # TSNE

	# if subset:
	# 	# df = df.iloc[np.random.choice(len(df), 2000, replace=False)]
	# 	df = df[(df['label'] == 0) | (df['label'] == 3) | (df['label'] == 5)]
	# 	# df = df[df['label'] > 0]

	# # X = df[map(str, range(512))].values
	# # X = (X - X.mean(axis=0)) / np.sqrt(X.var(axis=0))
	
	# # for i in range(512):
	# # 	df[str(i)] = X[:, i]

	# perp = 300

	# time_start = time.time()
	# tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=500)
	# tsne_results = tsne.fit_transform(df[map(str, range(512))].values)

	# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	# df['tsne-2d-one'] = tsne_results[:,0]
	# df['tsne-2d-two'] = tsne_results[:,1]


	# # TSNE 3D

	# if subset:
	# 	# df = df.iloc[np.random.choice(len(df), 2000, replace=False)]
	# 	df = df[(df['label'] == 0) | (df['label'] == 3) | (df['label'] == 5)]
	# 	# df = df[df['label'] > 0]

	# # X = df[map(str, range(512))].values
	# # X = (X - X.mean(axis=0)) / np.sqrt(X.var(axis=0))
	
	# # for i in range(512):
	# # 	df[str(i)] = X[:, i]

	# perp = 100

	# time_start = time.time()
	# tsne = TSNE(n_components=3, verbose=1, perplexity=perp, n_iter=500)
	# tsne_results = tsne.fit_transform(df[map(str, range(512))].values)

	# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	# df['tsne-3d-one'] = tsne_results[:,0]
	# df['tsne-3d-two'] = tsne_results[:,1]
	# df['tsne-3d-thr'] = tsne_results[:,2]


	K = 16 if 'gray' in df_path else 4

	# # KMeans

	# X = df[['tsne-2d-one', 'tsne-2d-two']].values

	# kmeans = KMeans(n_clusters=K).fit(X)

	# df['cluster'] = kmeans.labels_


	# # KMeans (High-D)

	# X = df[map(str, range(512))].values

	# # cluster_ids = df['cluster']
	# # centroids = np.array([X[cluster_ids == i].mean(axis=0) for i in range(K)])
	# # kmeans = KMeans(n_clusters=K, init=centroids).fit(X)

	# kmeans = KMeans(n_clusters=K).fit(X)

	# df['hd-cluster'] = kmeans.labels_


	# df.to_csv('bloodmnist-gray-all-unshuffled-tsne3d-clusters.csv', index=False)


	# # Plot

	# plt.figure(figsize=(5,5))

	# sns.scatterplot(
	# 	x="tsne-2d-one", y="tsne-2d-two",
	# 	hue="cluster", palette=sns.color_palette("bright", K),
	# 	data=df,
	# 	alpha=0.75
	# )

	# # plt.scatter(X[:, 0], X[:, 1], c=colors)

	# plt.title(f'{dataset} - clusters')
	# # plt.savefig(f'{dataset}{"-a" if subset else ""}.png')
	# plt.show()


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


	# plt.figure(figsize=(5,5))

	# sns.scatterplot(
	# 	x="tsne-2d-one", y="tsne-2d-two",
	# 	hue="colour-cluster", palette=sns.color_palette("bright", 4),
	# 	data=df,
	# 	alpha=0.75
	# )

	# # plt.scatter(X[:, 0], X[:, 1], c=colors)

	# plt.title(f'{dataset} - colour-clusters')
	# # plt.savefig(f'{dataset}{"-a" if subset else ""}.png')
	# plt.show()


	plt.figure(figsize=(5,5))

	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="label", palette=sns.color_palette("bright", num_classes - (5 if subset else 0)),
		data=df,
		alpha=0.75
	)

	# plt.scatter(X[:, 0], X[:, 1], c=colors)

	plt.title(f'{dataset} - labels')
	# plt.savefig(f'{dataset}{"-a" if subset else ""}.png')
	plt.show()


	plt.figure(figsize=(5,5))

	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="phase", palette=sns.color_palette("bright", 2),
		data=df,
		alpha=0.75
	)

	# plt.scatter(X[:, 0], X[:, 1], c=colors)

	plt.title(f'{dataset} - phases')
	# plt.savefig(f'{dataset}{"-a" if subset else ""}.png')
	plt.show()


	# # Plot 3D
	# df = df.iloc[np.random.choice(len(df), 1000, replace=False)]

	# fig = plt.figure(figsize=(15,5))

	# palette = ['blue', 'green', 'red', 'yellow', 'magenta', 'cyan', 'orange', 'brown']

	# # fig = plt.figure(figsize=(6,6))
	# # ax = Axes3D(fig, auto_add_to_figure=False)
	# # fig.add_axes(ax)
	# ax = fig.add_subplot(1, 3, 1, projection='3d')

	# # plot
	# sc = ax.scatter(
	# 	df['tsne-3d-one'],
	# 	df['tsne-3d-two'],
	# 	df['tsne-3d-thr'],
	# 	s = 40,
	# 	c = [palette[i] for i in df['label']],
	# 	alpha = 0.8
	# 	)

	# # legend
	# # plt.legend(, bbox_to_anchor=(1, 1), loc=2)

	# # plt.show()


	# palette = ['blue', 'green', 'red', 'yellow']

	# # fig = plt.figure(figsize=(6,6))
	# # ax = Axes3D(fig, auto_add_to_figure=False)
	# # fig.add_axes(ax)
	# ax = fig.add_subplot(1, 3, 2, projection='3d')

	# # plot
	# sc = ax.scatter(
	# 	df['tsne-3d-one'],
	# 	df['tsne-3d-two'],
	# 	df['tsne-3d-thr'],
	# 	s = 40,
	# 	c = [palette[i] for i in df['hd-cluster']],
	# 	alpha = 0.8
	# 	)

	# # legend
	# # plt.legend(, bbox_to_anchor=(1, 1), loc=2)

	# # plt.show()


	# palette = {'train': 'blue', 'test': 'red'}

	# # fig = plt.figure(figsize=(6,6))
	# # ax = Axes3D(fig, auto_add_to_figure=False)
	# # fig.add_axes(ax)
	# ax = fig.add_subplot(1, 3, 3, projection='3d')

	# # plot
	# sc = ax.scatter(
	# 	df['tsne-3d-one'],
	# 	df['tsne-3d-two'],
	# 	df['tsne-3d-thr'],
	# 	s = 40,
	# 	c = [palette[i] for i in df['phase']],
	# 	alpha = 0.8
	# 	)

	# # legend
	# # plt.legend(, bbox_to_anchor=(1, 1), loc=2)

	# plt.show()



	# [11323,  9669, 12812,  4770]
	# [11323,  9669, (test)853,  4770]
