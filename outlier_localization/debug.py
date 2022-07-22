import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	
	dataset = 'retinamnist'
	model = 'cpc'
	suffix = '_'.join(['healthy', 'patchwise', 'loss'])
	num_classes = 5

	crop = '24-0'
	encoder, D = 'resnet18', 512
	norm = 'layer'
	grid_size = 5
	pred_directions = 4
	cpc_patch_aug = 'True'
	gray = '_colour'
	model_num = '200'

	df_path = f'../TrainedModels/{dataset}/trained_{model}_{encoder}_crop{crop}{gray}_grid{grid_size}_{norm}Norm_{pred_directions}dir_aug{cpc_patch_aug}_{model_num}{dataset}_{suffix}.csv'
	
	df = pd.read_csv(df_path)
	df = df[df['phase'] == 'test']


	patchwise_loss = df[[f'{i},{j}' for i in range(grid_size) for j in range(grid_size)]].values.reshape((len(df), grid_size, grid_size))

	mean_loss = patchwise_loss.mean(axis = (1, 2))
	max_loss = patchwise_loss.max(axis = (1, 2))
	median_loss = np.median(patchwise_loss, axis = (1, 2))
	class_loss = [mean_loss[df['label'] == c].mean() for c in range(num_classes)]

	labels = df['label'].values
	labels = labels > 0


	''' total loss threshold

	fpr, tpr, thresholds = roc_curve(labels > 0, mean_loss)
	auc = roc_auc_score(labels > 0, mean_loss)

	print(auc)

	plt.plot(fpr, tpr)
	plt.plot(fpr, fpr)
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title('ROC - CPC Loss Threshold Classifier')
	plt.legend(['classifier', 'random'])
	plt.show()

	# '''
	

	# ''' patchwise loss threshold, number of outlier patches

	patch_threshold = np.linspace(patchwise_loss.min(), patchwise_loss.max() / 5, 25)
	outlier_count_threshold = np.arange(grid_size ** 2)

	outlier_count = (patchwise_loss[..., None] > patch_threshold).sum(axis = (1, 2))
	prediction = outlier_count[..., None] > outlier_count_threshold

	acc = (labels[:, None, None] == prediction).mean(axis = 0)

	pt, ot = np.unravel_index(acc.argmax(), acc.shape)

	print(acc[pt, ot])

	# plt.imshow(acc)
	# plt.title('acc')

	plt.figure(figsize=(10,8))

	ax = sns.heatmap(acc, xticklabels = outlier_count_threshold / grid_size ** 2, yticklabels = patch_threshold.round(3))
	ax.set_ylabel('patchwise loss threshold')
	ax.set_xlabel('outlier-patch fraction threshold')

	plt.show()

	# '''
	

	''' patchwise loss (sample-normalized) threshold, number of outlier patches

	# patchwise_loss = (patchwise_loss - patchwise_loss.mean()) / np.sqrt(patchwise_loss.var())
	# patchwise_loss = patchwise_loss / np.sqrt(patchwise_loss.var(0))

	patch_threshold = np.linspace(patchwise_loss.min(), patchwise_loss.max()/5, 25)
	outlier_count_threshold = np.arange(grid_size ** 2)

	outlier_count = (patchwise_loss[..., None] > patch_threshold).sum(axis = (1, 2))
	prediction = outlier_count[..., None] > outlier_count_threshold

	acc = (labels[:, None, None] == prediction).mean(axis = 0)

	pt, ot = np.unravel_index(acc.argmax(), acc.shape)

	print('patchwise loss threshold:', patch_threshold[pt])
	print('outlier count threshold: ', outlier_count_threshold[ot])
	print('accuracy:', acc[pt, ot])

	plt.imshow(acc)
	plt.title('acc')
	plt.show()

	# '''
	

	''' linear regression on patchwise losses

	patchwise_loss = patchwise_loss.reshape((-1, grid_size ** 2))

	# reg = LinearRegression()
	reg = LogisticRegression()
	reg.fit(patchwise_loss, labels)

	# print(reg.score(patchwise_loss.reshape((-1, grid_size ** 2)), labels))

	pred = reg.predict(patchwise_loss)
	# pred -= pred.min()
	# pred *= 4 / pred.max()

	C = confusion_matrix(labels, pred.round())

	# '''
