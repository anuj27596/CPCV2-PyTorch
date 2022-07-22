import pandas as pd

import medmnist


class BloodMNISTClusters(medmnist.dataset.BloodMNIST):

	def __init__(self, split, cluster_data_path, cluster_column, transform=None, target_transform=None, download=False, as_rgb=False, root=medmnist.info.DEFAULT_ROOT):

		super().__init__(split, transform, target_transform, download, as_rgb, root)

		df = pd.read_csv(cluster_data_path)
		df = df[df['phase'] == split]

		self.cluster_array = df[cluster_column].to_numpy()


	def __getitem__(self, index):

		img, target = super().__getitem__(index)
		cluster = self.cluster_array[index]

		return img, cluster
