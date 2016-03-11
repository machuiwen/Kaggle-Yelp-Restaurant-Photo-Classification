data_path = '/mnt/data/test_photos/'

f = open('/mnt/data/test_photo_to_biz.csv', 'r')
f.readline()
fo = open('/mnt/data/kaggle_test_images.csv', 'w')
for line in f:
    photo_id, biz_id = line.strip().split(',')
    fo.write(data_path + photo_id + '.jpg' + '\n')
    
f.close()
fo.close()
