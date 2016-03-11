import matplotlib.pyplot as plt
import numpy as np
import os, sys
from sklearn import metrics

npzfile = np.load(sys.argv[1])
label_array, score_array = npzfile['arr_0'], npzfile['arr_1']
prediction_array = score_array
print np.mean(metrics.f1_score(label_array, prediction_array, average=None)), '********'
# score_array = 1 / (1 + np.exp(-score_array))
# prediction_array = np.zeros_like(score_array, dtype='int64')
# threshold = 0.3
# prediction_array[score_array >= threshold] = 1
# prediction_array[score_array < threshold] = 0
accuracy = np.sum(prediction_array == label_array, axis=0) / float(label_array.shape[0])
# print accuracy
print np.mean(accuracy)

num_labels = 9
aucs = np.zeros(num_labels)
for i in range(num_labels):
	fpr, tpr, _ = metrics.roc_curve(label_array[:, i], score_array[:, i], pos_label=1)
	aucs[i] = metrics.auc(fpr, tpr)
	resolution = 1000
	samples = np.array(range(resolution + 1)) / float(resolution)
	plt.subplot(3, 3, i + 1)
	plt.plot(fpr, tpr)
	plt.plot(samples, samples)
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
f1_scores = metrics.f1_score(label_array, prediction_array, average=None)
# print f1_scores
print np.mean(f1_scores)
print aucs - 0.5
plt.show()
