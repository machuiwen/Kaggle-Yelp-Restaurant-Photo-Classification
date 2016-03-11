caffe_root = '/home/ubuntu/caffe/' 
data_root = '/mnt/data/'
stdout = '/home/ubuntu/machuiwen_output.txt'

import h5py
import numpy as np
import sys
import caffe
import os
import pandas as pd

sys.path.insert(0, caffe_root + 'python')
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    # !caffe_root/scripts/download_model_binary.py ../models/bvlc_reference_caffenet
    
## Use GPU    
caffe.set_device(0)
caffe.set_mode_gpu()

def extract_features(images, layer = 'fc7'):
    print "=============== extracting features for one batch ================"
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]]

    num_images= len(images)
    net.blobs['data'].reshape(num_images,3,227,227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data

def extract_dataset(output_path, filelist_path):
    print "=============== setting output file ================"
    # Initialize files
    # extract image features and save it to .h5
    f = h5py.File(output_path,'w')
    filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
    feature = f.create_dataset('feature',(0,4096), maxshape = (None,4096))
    f.close()

    print "=============== generating file list ================"
    f_photos = open(filelist_path, 'r')
    images_list = [line.strip().split(' ')[0] for line in f_photos]
    f_photos.close()

    num_images = len(images_list)
    print "Number of images: ", num_images
    batch_size = 500

    for i in range(0, num_images, batch_size): 
        images = images_list[i: min(i+batch_size, num_images)]
        features = extract_features(images, layer='fc7')
        num_done = i+features.shape[0]
        f= h5py.File(output_path,'r+')
        f['photo_id'].resize((num_done,))
        f['photo_id'][i: num_done] = np.array(images)
        f['feature'].resize((num_done,features.shape[1]))
        f['feature'][i: num_done, :] = features
        f.close()
        print "=============== # images processed: ", num_done, " ================"
        tempfile = open(stdout, 'w')
        tempfile.write("Number of images processed: %d, total %d." % (num_done, num_images))
        tempfile.close()

def extract_kaggletest(output_path, filelist_path):
    batch_size = 500
    print "=============== setting output file ================"
    f = h5py.File(output_path, 'w')
    filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
    feature = f.create_dataset('feature',(0,4096), maxshape = (None,4096))
    f.close()

    print "=============== generating file list ================"
    test_photos = pd.read_csv(filelist_path)
    test_folder = data_root+'test_photos/'
    test_images = [os.path.join(test_folder, str(x)+'.jpg') for x in test_photos['photo_id'].unique()]
    num_test = len(test_images)
    print "Number of test images: ", num_test
    # Test Images
    for i in range(0, num_test, batch_size):
        images = test_images[i: min(i+batch_size, num_test)]
        features = extract_features(images, layer='fc7')
        num_done = i+features.shape[0]

        f= h5py.File(output_path,'r+')
        f['photo_id'].resize((num_done,))
        f['photo_id'][i: num_done] = np.array(images)
        f['feature'].resize((num_done,features.shape[1]))
        f['feature'][i: num_done, :] = features
        f.close()
        print "=============== # Test images processed: ", num_done, " ================"
        tempfile = open(stdout, 'w')
        tempfile.write("Kaggle Test - Number of images processed: %d, total %d." % (num_done, num_test))
        tempfile.close()

## Extract features from training image

train_output = data_root+'caffenet_train_image_fc7features.h5'
train_list = data_root+'split/'+'train_images_100k.txt'

# extract_dataset(train_output, train_list)

## Extract features from test image

test_output = data_root+'caffenet_test_image_fc7features.h5'
test_list = data_root+'split/'+'test_images_100k.txt'

# extract_dataset(test_output, test_list)

kaggletest_output = data_root+'caffenet_kaggletest_image_fc7features.h5'
kaggletest_list = data_root+'test_photo_to_biz.csv'

extract_kaggletest(kaggletest_output, kaggletest_list)
