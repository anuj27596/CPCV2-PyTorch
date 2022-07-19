import numpy as np
import pandas as pd

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


	global_mean_cpc = cpcvs.mean(axis = 0)
	cluster_mean_cpc = [cpcvs[cluster_ids == i].mean(axis = 0) for i in range(4)]
	# cluster_mean_cpc = cpcvs[[11323,  9669, 12812,  4770]]

	global_mean_img = images.mean(axis = 0)
	cluster_mean_img = [images[cluster_ids == i].mean(axis = 0) for i in range(4)]
	# cluster_mean_img = images[[11323,  9669, 12812,  4770]]


	# plt.imshow(global_mean_img)
	# plt.savefig('global_mean.png')

	print('--- images ---')
	print('global std  ', images.std())

	for i in range(4):

		print(f'clus-{i} dst  ', ((cluster_mean_img[i] - global_mean_img) ** 2).mean() ** 0.5)
		# print(f'clus-{i} dst  ', np.linalg.norm(cluster_mean_img[i] - global_mean_img))
		print(f'clus-{i} std  ', images[cluster_ids == i].std())

		# f = np.abs(cluster_mean_img[i] - global_mean_img).max()
		# # f = 3 * (cluster_mean_img[i] - global_mean_img).std()

		# # plt.imshow(cluster_mean_img[i])
		# # plt.savefig(f'cluster_{i}_mean.png')
		
		# plt.imshow(((cluster_mean_img[i] - global_mean_img) / f + 1) / 2)
		# plt.savefig(f'cluster_{i}_mean_delta.png')

	print('--- cpc vectors ---')
	print('global std  ', cpcvs.std())

	for i in range(4):

		print(f'clus-{i} dst  ', ((cluster_mean_cpc[i] - global_mean_cpc) ** 2).mean() ** 0.5)
		# print(f'clus-{i} dst  ', np.linalg.norm(cluster_mean_cpc[i] - global_mean_cpc))
		print(f'clus-{i} std  ', cpcvs[cluster_ids == i].std())
