import os, shutil
import cv2
import numpy as np

def hash(shape):
    return str(shape[0] + shape[1])

def detect_duplicate():
    folder = '../train_photos/'
    inputfile = open('biz_ids.txt', 'r')


    count = 0
    for line in inputfile:
        if count % 100 == 0:
            print count, '/', ' 2,000'
        count += 1
        biz_id = line.strip()
        if os.path.isdir(folder + biz_id):
            mypath = folder + biz_id
            files = os.listdir(folder + biz_id)[1:]
            images = {}
            for f in files:
                imagepath = mypath + '/' + f
                img = cv2.imread(imagepath)
                key = hash(img.shape)
                if not key in images:
                    images[key] = []
                dup = False
                for item in images[key]:
                    if np.array_equal(item, img):
                        dup = True
                        print imagepath
                        break
                if not dup:
                    images[key].append(img)
    inputfile.close()

def remove_dup():
    path = '../train_photos/'
    dupfile = open('train_duplicate.csv', 'r')
    photo2bizfile = open('train_photo_to_biz_ids.csv', 'r')
    photo2bizfile_new = open('train_photo_uniq_to_biz_ids.csv', 'w')
    photo2bizfile_new.write(photo2bizfile.readline())
    photoid_bizid = {}
    for line in photo2bizfile:
        photo_id, biz_id = line.strip().split(',')
        photoid_bizid[photo_id] = biz_id
    dupfile.readline()
    for line in dupfile:
        biz_id, photo_id = line.strip().split(',')
        photopath = path + biz_id + '/' + photo_id + '.jpg'
        os.remove(photopath)
        del photoid_bizid[photo_id]
    for key in photoid_bizid:
        photo2bizfile_new.write(key + ',' + photoid_bizid[key] + '\n')
    dupfile.close()
    photo2bizfile.close()
    photo2bizfile_new.close()

remove_dup()