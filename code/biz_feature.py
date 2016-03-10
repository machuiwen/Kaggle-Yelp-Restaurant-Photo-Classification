import numpy as np
import pandas as pd 
import h5py
import time

data_root = '/mnt/data/'
train_features_path = data_root+'caffenet_train_image_fc7features.h5'
train_biz_features_path = data_root+'caffenet_train_biz_fc7features.csv'
train_list = data_root+'split/'+'train_images_100k.txt'

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
    # labels = pd.read_csv(data_root+'train.csv').dropna()
    # labels['labels'] = labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
    # labels.set_index('business_id', inplace=True)
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
        # label = labels.loc[biz]['labels']
        image_index = []
        for i, item in enumerate(images_list):
            if item[0] == biz:
                image_index.append(i)
        folder = data_root+'train_photos/'
        
        # sanity check
        if len(image_index) == 0:
            print "***** no image for biz", biz, " *****"
        features = train_image_features[image_index]
        mean_feature = list(np.mean(features,axis=0))

        df.loc[count] = [biz, label, mean_feature]
        count += 1
        if count % 100 == 0:
            print "===== Buisness processed: ", count, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec", " ====="

    with open(biz_feat_path,'w') as f:  
        df.to_csv(f, index=False)

compute_biz_features(train_features_path, train_biz_features_path, train_list)
