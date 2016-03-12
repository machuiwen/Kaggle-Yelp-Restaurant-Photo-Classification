import numpy as np
import pandas as pd 
import h5py
import time

data_root = '/mnt/data/'
model_name = 'vgg'
train_features_path = data_root+model_name+'_train_image_fc7features.h5'
train_biz_features_path = data_root+model_name+'_train_biz_fc7features.csv'
train_list = data_root+'split/'+'train_images_100k.txt'
test_features_path = data_root+model_name+'_test_image_fc7features.h5'
test_biz_features_path = data_root+model_name+'_test_biz_fc7features.csv'
test_list = data_root+'split/'+'test_images_100k.txt'
kaggle_features_path = data_root+model_name+'_kaggletest_image_fc7features.h5'
kaggle_biz_features_path = data_root+model_name+'_kaggletest_biz_fc7features.csv'

def compute_biz_features(feat_path, biz_feat_path, filelist_path):
    f_photos = open(filelist_path, 'r')
    images_list = [[int(x) for x in line.split(' ')[0][23:-4].split('/')] for line in f_photos] # biz - image
    f_photos.close()
    f_labels = open(data_root+'train.csv', 'r')
    f_labels.readline()
    labels = {}
    for line in f_labels:
        biz, l = line.strip().split(',')
        labels[int(biz)] = [int(x) for x in l.split()]
    f_labels.close()
    biz_ids = set()
    for item in images_list:
        biz_ids.add(item[0])
    print "===== Number of business: ", len(biz_ids), " ====="

    ## Load image features
    f = h5py.File(feat_path,'r')
    train_image_features = np.copy(f['feature'])
    f.close()

    t = time.time()
    ## For each business, compute a feature vector 
    df = pd.DataFrame(columns=['business','label','feature vector'])
    count = 0
    for biz in biz_ids:
        label = tuple(labels[biz])
        image_index = []
        for i, item in enumerate(images_list):
            if item[0] == biz:
                image_index.append(i)
        folder = data_root+'train_photos/'
        features = train_image_features[image_index]
        mean_feature = list(np.mean(features,axis=0))

        df.loc[count] = [biz, label, mean_feature]
        count += 1
        if count % 100 == 0:
            print "===== Buisness processed: ", count, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec", " ====="

    with open(biz_feat_path,'w') as f:  
        df.to_csv(f, index=False)

def kaggletest_biz_features(feat_path, biz_feat_path):
    test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
    biz_ids = test_photo_to_biz['business_id'].unique()
    ## Load image features
    f = h5py.File(feat_path,'r')
    image_filenames = list(np.copy(f['photo_id']))
    image_filenames = [name.split('/')[-1][:-4] for name in image_filenames]  #remove the full path and the str ".jpg"
    image_features = np.copy(f['feature'])
    f.close()
    print "===== Number of business: ", len(biz_ids), " ====="
    
    df = pd.DataFrame(columns=['business','feature vector'])
    count = 0
    t = time.time()
    for biz in biz_ids:
        image_ids = test_photo_to_biz[test_photo_to_biz['business_id']==biz]['photo_id'].tolist()  
        image_index = [image_filenames.index(str(x)) for x in image_ids]         
        folder = data_root+'test_photos/'   
        features = image_features[image_index]
        mean_feature =list(np.mean(features,axis=0))

        df.loc[count] = [biz, mean_feature]
        count += 1
        if count % 100 == 0:
            print "===== Buisness processed: ", count, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec", " ====="
   
    with open(biz_feat_path,'w') as f:
        df.to_csv(f, index=False)
# compute_biz_features(train_features_path, train_biz_features_path, train_list)
# compute_biz_features(test_features_path, test_biz_features_path, test_list)
kaggletest_biz_features(kaggle_features_path, kaggle_biz_features_path)
