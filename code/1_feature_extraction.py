caffe_root = '/home/ubuntu/caffe/' 
data_root = '/mnt/data/'
stdout = '/home/ubuntu/machuiwen_output.txt'

model_name = 'google'
if model_name == 'caffenet':
    prototxt_path = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_path = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    input_width, input_height = 227, 227
    batch_sz = 512
    feature_dim = 4096
    layer_name = 'fc7'
elif model_name == 'vgg':
    prototxt_path = caffe_root + 'models/VGG/VGG_CNN_S_deploy.prototxt'
    model_path = caffe_root + 'models/VGG/VGG_CNN_S.caffemodel'
    mean_path = caffe_root + 'models/VGG/VGG_mean.npy'
    input_width, input_height = 224, 224
    batch_sz = 256
    feature_dim = 4096
    layer_name = 'fc7'
elif model_name == 'google':
    prototxt_path = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
    model_path = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy' # we only use channel avg - [104, 117, 123]
    input_width, input_height = 224, 224
    batch_sz = 118
    feature_dim = 1024
    layer_name = 'pool5/7x7_s1'


import h5py
import numpy as np
import sys
import caffe
import os
import pandas as pd

## Download model
#sys.path.insert(0, caffe_root + 'python')
#if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#    print("Downloading pre-trained CaffeNet model...")
#    # !caffe_root/scripts/download_model_binary.py ../models/bvlc_reference_caffenet
    
## Use GPU    
caffe.set_device(0)
caffe.set_mode_gpu()

def extract_features(images, layer='fc7'):
    print "=============== extracting features for one batch ================"
    net = caffe.Net(prototxt_path, model_path, caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(mean_path).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]]

    num_images= len(images)
    net.blobs['data'].reshape(num_images,3,input_width,input_height)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()
    feature = net.blobs[layer].data
    return feature.reshape(feature.shape[0:2])

def extract_dataset(output_path, filelist_path):
    print "=============== setting output file ================"
    # Initialize files
    # extract image features and save it to .h5
    f = h5py.File(output_path,'w')
    filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
    feature = f.create_dataset('feature',(0,feature_dim), maxshape = (None,feature_dim))
    f.close()

    print "=============== generating file list ================"
    f_photos = open(filelist_path, 'r')
    images_list = [line.strip().split(' ')[0] for line in f_photos]
    f_photos.close()

    num_images = len(images_list)
    print "Number of images: ", num_images
    batch_size = batch_sz

    for i in range(0, num_images, batch_size): 
        images = images_list[i: min(i+batch_size, num_images)]
        features = extract_features(images, layer=layer_name)
        num_done = i+features.shape[0]
        f= h5py.File(output_path,'r+')
        f['photo_id'].resize((num_done,))
        f['photo_id'][i: num_done] = np.array(images)
        f['feature'].resize((num_done,features.shape[1]))
        f['feature'][i: num_done, :] = features
        f.close()
        print "=============== # images processed: ", num_done, " ================"
        tempfile = open(stdout, 'w')
        tempfile.write(filelist_path[16:-4] + " Number of images processed: %d, total %d." % (num_done, num_images))
        tempfile.close()

def extract_kaggletest(output_path, filelist_path):
    batch_size = batch_sz
    print "=============== setting output file ================"
    f = h5py.File(output_path, 'w')
    filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
    feature = f.create_dataset('feature',(0,feature_dim), maxshape = (None,feature_dim))
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
        features = extract_features(images, layer=layer_name)
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

train_output = data_root+model_name+'_train_image_fc7features.h5'
train_list = data_root+'split/'+'train_images_100k.txt'

extract_dataset(train_output, train_list)

## Extract features from test image

test_output = data_root+model_name+'_test_image_fc7features.h5'
test_list = data_root+'split/'+'test_images_100k.txt'

extract_dataset(test_output, test_list)

kaggletest_output = data_root+model_name+'_kaggletest_image_fc7features.h5'
kaggletest_list = data_root+'test_photo_to_biz.csv'

extract_kaggletest(kaggletest_output, kaggletest_list)
