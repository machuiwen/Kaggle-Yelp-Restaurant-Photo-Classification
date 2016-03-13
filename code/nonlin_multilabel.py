import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import time

## Running Mode
# 0: use kaggle train / test split
# 1: only use kaggle train for train / test
# 2: only use 1000 biz for train / test
mode = 0
model_name = 'google'
n_process = -1
print "===== ", model_name, " ====="

## Load Data
data_root = '/mnt/data/'
train_biz_features = data_root + model_name + "_train_biz_fc7features.csv"
test_biz_features = data_root + model_name + "_test_biz_fc7features.csv"
kaggle_biz_features = data_root + model_name + "_kaggletest_biz_fc7features.csv"
submission_file = data_root + "submission_" + model_name + "_fc7_rbf.csv"
roc_file = '/home/ubuntu/caffe/tmpdata/RBFSVM_'+model_name+'_roc.npz'

train_df = pd.read_csv(train_biz_features)
X_train = train_df['feature vector'].values
y_train = train_df['label'].values

if mode == 0 or mode == 1:
    test_df = pd.read_csv(test_biz_features)
    X_test = test_df['feature vector'].values
    y_test = test_df['label'].values

if mode == 0:
    kaggle_df = pd.read_csv(kaggle_biz_features)
    X_kaggle = kaggle_df['feature vector'].values

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

if mode == 0 or mode == 1:
    X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])
    y_test = np.array([convert_label_to_array(y) for y in test_df['label']])

if mode == 0:
    X_kaggle = np.array([convert_feature_to_vector(x) for x in kaggle_df['feature vector']])

# According to mode rearrange dataset
if mode == 0:
    X_train = np.vstack((X_train, X_test))
    y_train = np.hstack((y_train, y_test))
    X_test = X_kaggle
    y_test = None
elif mode == 2:
    half = X_train.shape[0] / 2
    X_test = X_train[half:,]
    y_test = y_train[half:]
    X_train = X_train[:half,]
    y_train = y_train[:half]
# Now X_train is the training dataset (including validation set for tuning hyperparameters)
# X_test is the test dataset, for mode 0, y_test = None, for other modes, we can compute test
# F1 score and test accuracy.

## Hyperparameter tuning
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
# caffenet: C = 0.01 best
best_f1 = -1
best_c = None
best_classifier = None
best_predict = None
for c in np.logspace(0,2,21):
    t = time.time()
    classifier = OneVsRestClassifier(svm.SVC(C=c, kernel='rbf', probability=True, \
        random_state=random_state), n_jobs=n_process)
    print "===== training svm for C =", c, " ====="
    classifier.fit(X_ptrain, y_ptrain)
    print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
    print "===== predicting on validation set ====="
    y_pval_predict = classifier.predict(X_pval)

    # Compute F1 score on validation set
    print "===== computing f1 score on validation set ====="
    f1_scores = f1_score(y_pval, y_pval_predict, average=None)
    f1 = np.mean(f1_scores)
    print "F1 score: ", f1
    print "Individual Class F1 score: ", f1_scores

    if f1 > best_f1:
        best_f1 = f1
        best_c = c
        best_classifier = classifier
        best_predict = y_pval_predict

print "Best C:", best_c, "Best mean f1 score:", best_f1
with open("/home/ubuntu/best_rbf_svm.txt", 'a') as result:
    result.write("----- Experiment -----\n")
    result.write("Mode: " + str(mode) + '\n')
    result.write("Model: " + model_name + '\n')
    result.write("Best C: " + str(best_c) + '\n')
    result.write("Best mean f1: " + str(best_f1) + '\n')

# Data statistics
print "===== showing predicting statistics on validation set ====="
statistics = pd.DataFrame(columns=["attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
statistics.loc["biz count"] = np.append(np.sum(best_predict, axis=0), len(best_predict))
pd.options.display.float_format = '{:.0f}%'.format
statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(best_predict) 
print statistics


## Re-Train a SVM using all training data, and make predictions on test set

t = time.time()
y_train = mlb.fit_transform(y_train)
if y_test is not None:
    y_test = mlb.fit_transform(y_test)
classifier = OneVsRestClassifier(svm.SVC(C=best_c, kernel='rbf', probability=True, \
    random_state=random_state), n_jobs=n_process)

print "===== training svm on all training data ====="
classifier.fit(X_train, y_train)
print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
print "===== predicting on test set ====="
y_test_predict = classifier.predict(X_test)
y_predict_label = mlb.inverse_transform(y_test_predict)

print "===== showing predicting statistics on test set ====="
statistics = pd.DataFrame(columns=["attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
statistics.loc["biz count"] = np.append(np.sum(y_test_predict, axis=0), len(y_test_predict))
pd.options.display.float_format = '{:.0f}%'.format
statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_test_predict) 
print statistics

if y_test is not None:
    print "===== computing f1 score on test set ====="
    f1_scores = f1_score(y_test, y_test_predict, average=None)
    f1 = np.mean(f1_scores)
    print "##### F1 score: ", f1
    print "##### Individual Class F1 score: ", f1_scores
    # Save ouput
    with open("/home/ubuntu/best_rbf_svm.txt", 'a') as result:
        result.write('Testset f1: ' + str(f1) + '\n')
    label_array = y_test
    prediction_array = classifier.predict_proba(X_test)
    np.savez(roc_file, label_array, prediction_array)
else:
    # generate submission
    test_data_frame  = pd.read_csv(kaggle_biz_features)
    df = pd.DataFrame(columns=['business_id','labels'])
    for i in range(len(test_data_frame)):
        biz = test_data_frame.loc[i]['business']
        label = y_predict_label[i]
        label = str(label)[1:-1].replace(",", "")
        df.loc[i] = [str(biz), label]

    with open(submission_file,'w') as f:
        df.to_csv(f, index=False) 
