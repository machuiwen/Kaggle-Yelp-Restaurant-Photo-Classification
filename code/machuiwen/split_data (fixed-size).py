import random

def make_data_input(biz_ids, output):
	fout = open(output, 'w')
	for biz in biz_ids:
		for photo in d[biz]:
			photo_path = path + biz + '/' + photo + '.jpg'
			labels = biz_label[biz]
			fout.write(photo_path + ' ' + str(labels) + '\n')
	fout.close()

f = open('uniq_train_photo_to_biz_ids.csv', 'r')
d = {} # biz_id - photo_ids
for line in f:
	photo, biz = line.strip().split(',')
	if not biz in d:
		d[biz] = []
	d[biz].append(photo)
f.close()

keys = d.keys()
random.shuffle(keys)
count = 0
total = 500
train_keys = []
test_keys = []
for key in keys:
	if count < total:
		count += len(d[key])
		train_keys.append(key)
	elif 2 * total > count >= total:
		count += len(d[key])
		test_keys.append(key)
	else:
		break

path = '/mnt/data/train_photos/'
fbiz_label = open("train.csv", 'r')
fbiz_label.readline()
biz_label = {}
for line in fbiz_label:
    biz, labels = line.strip().split(',')
    tgt = 0
    labels = labels.split()
    for i in range(9):
        tgt = tgt << 1
        if str(i) in labels:
            tgt = tgt + 1
    biz_label[biz] = tgt

train_dir = 'train_images_500.txt'
test_dir = 'test_images_500.txt'
make_data_input(train_keys, train_dir)
make_data_input(test_keys, test_dir)
