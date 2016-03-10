/home/ubuntu/caffe/build/tools/convert_imageset -check_size -resize_height 227 -resize_width 227 -shuffle /. /mnt/data/train_images_rand_10k.txt /mnt/data/train_lmdb
/home/ubuntu/caffe/build/tools/compute_image_mean /mnt/data/train_lmdb/ /mnt/data/yelp_mean_227.binaryproto
rm -rf /mnt/data/train_lmdb
