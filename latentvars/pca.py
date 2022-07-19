import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import medmnist

import matplotlib.pyplot as plt


if __name__ == '__main__':
	
	df_path = 'bloodmnist-all-unshuffled-tsne-clusters.csv'
	df = pd.read_csv(df_path)

	train_dataset = medmnist.dataset.BloodMNIST(root = '../data/bloodmnist', split = 'train')
	test_dataset = medmnist.dataset.BloodMNIST(root = '../data/bloodmnist', split = 'test')


	images = np.stack([np.array(img) for img, lbl in (train_dataset + test_dataset)]) / 255

	cpcvs = df[map(str, range(512))].values

	cluster_ids = df['hd-cluster'].values

	

	# n_pca = 6

	# n_feat = 28*28*3

	# global_pca = PCA(n_components = n_pca).fit(images.reshape((-1, n_feat)))

	# cluster_pca = [PCA(n_components = n_pca).fit(images[cluster_ids == i].reshape((-1, n_feat))) for i in range(4)]

	# global_pax = global_pca.components_

	# cluster_pax = [pca.components_ for pca in cluster_pca]

	# for i in range(4):
	# 	for j in range(n_pca):
	# 		print(round(global_pax[j] @ cluster_pax[i][j], 4), end='\t')
	# 	print()
