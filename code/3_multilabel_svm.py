import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import time

## Load Data
data_root = '/mnt/data/'
n_process = -2
train_biz_features = data_root + "caffenet_train_biz_fc7features.csv"
test_biz_features = data_root + "caffenet_test_biz_fc7features.csv"

train_df = pd.read_csv(train_biz_features)
# test_df = pd.read_csv(test_biz_features)

X_train = train_df['feature vector'].values
y_train = train_df['label'].values
# X_test = test_df['feature vector'].values
# y_test = test_df['label'].values

def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x)>0]

def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]

X_train = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
# X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])
# y_test = np.array([convert_label_to_array(y) for y in test_df['label']])

## Train a SVM using 80% training data, and assess performance (F1-score)

# Convert list of labels to binary matrix
# Transform between iterable of iterables and a multilabel format
mlb = MultiLabelBinarizer()
y_ptrain = mlb.fit_transform(y_train)
random_state = np.random.RandomState(425)
# Split data
X_ptrain, X_pval, y_ptrain, y_pval = train_test_split(X_train, y_ptrain, test_size=.2, \
    random_state = random_state)
# Tuning hyperparameter C for SVM
# range np.logspace(-3, 2, 6)
# C = 0.01 best
best_f1 = -1
best_c = None
best_classifier = None
for c in [0.01]:
    t = time.time()
    classifier = OneVsRestClassifier(svm.SVC(C=c, kernel='linear', probability=True, \
        random_state=random_state), n_jobs=n_process)
    print "===== training svm for C =", c, " ====="
    classifier.fit(X_ptrain, y_ptrain)
    print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
    print "===== predicting on validation set ====="
    y_pval_predict = classifier.predict(X_pval)

    # Compute F1 score on validation set
    print "===== computing f1 score on validation set ====="
    f1 = f1_score(y_pval, y_pval_predict, average='micro')
    print "F1 score: ", f1
    f1_scores = f1_score(y_pval, y_pval_predict, average=None)
    print "Individual Class F1 score: ", f1_scores
    print np.sum(np.int64(classifier.predict_proba(X_pval) == 0.5))
    

    if f1 > best_f1:
        best_f1 = f1
        best_c = c
        best_classifier = classifier

print "Best C:", best_c, "Best mean f1 score:", best_f1
label_array = y_pval
prediction_array = best_classifier.predict(X_pval)
print np.mean(f1_score(label_array, prediction_array, average=None))
np.savez('/home/ubuntu/caffe/tmpdata/SVM_roc.npz', label_array, prediction_array)

# # Data statistics
# print "===== showing predicting statistics on validation set ====="
# statistics = pd.DataFrame(columns=["attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
# statistics.loc["biz count"] = np.append(np.sum(y_pval_predict, axis=0), len(y_pval_predict))
# pd.options.display.float_format = '{:.0f}%'.format
# statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_pval_predict) 
# print statistics


## Re-Train a SVM using all training data, and make predictions on test set

# t = time.time()
# y_train = mlb.fit_transform(y_train)
# y_test = mlb.fit_transform(y_test)
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True), n_jobs=n_process)

# print "===== training svm on all training data ====="
# classifier.fit(X_train, y_train)
# print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

# print "===== predicting on test set ====="
# y_test_predict = classifier.predict(X_test)

# print "===== showing predicting statistics on test set ====="
# statistics = pd.DataFrame(columns=["attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
# statistics.loc["biz count"] = np.append(np.sum(y_test_predict, axis=0), len(y_test_predict))
# pd.options.display.float_format = '{:.0f}%'.format
# statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_test_predict) 
# print statistics

# print "===== computing f1 score on test set ====="
# print "F1 score: ", f1_score(y_test, y_test_predict, average='micro')
# print "Individual Class F1 score: ", f1_score(y_test, y_test_predict, average=None)
