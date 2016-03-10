# biz - labels

inputfile = open('train.csv', 'r')

label_text = [
	'good for lunch         ',
	'good for dinner        ',
	'take reservations      ',
	'outdoor seating        ',
	'restaurant is expensive',
	'has alcohol            ',
	'has table service      ',
	'ambience is classy     ',
	'good for kids          '
]
inputfile.readline()
biz_total = 0
label_total = 0
label_count = [0] * 9
min_label = 100000
max_label = 0
labels_dict = {}
for line in inputfile:
	biz_total += 1
	biz_id, labels = line.strip().split(',')
	label_ids = [int(x) for x in labels.split()]
	labels_dict[biz_id] = len(label_ids)
	label_total += len(label_ids)
	min_label = min(min_label, len(label_ids))
	max_label = max(max_label, len(label_ids))
	for l in label_ids:
		label_count[l] += 1

inputfile.close()

print 'average #labels per biz: ', float(label_total) / biz_total
print 'max #labels for one biz: ', max_label
print 'min #labels for one biz: ', min_label
for i in range(9):
	print 'label: ', i, label_text[i], 'percent: ', str(float(label_count[i]) / biz_total * 100)


# biz - # images

inputfile = open('train_photo_to_biz_ids.csv', 'r')
biz_photo_dict = {}
inputfile.readline()
for line in inputfile:
	photo_id, biz_id = line.strip().split(',')
	if not biz_id in biz_photo_dict:
		biz_photo_dict[biz_id] = []
	biz_photo_dict[biz_id].append(photo_id)

biz_total = len(biz_photo_dict)
photo_total = 234842.0
min_photo = 234842
max_photo = -1
num_photos_hist = {}
for key in biz_photo_dict:
	min_photo = min(min_photo, len(biz_photo_dict[key]))
	max_photo = max(max_photo, len(biz_photo_dict[key]))
	if not len(biz_photo_dict[key]) in num_photos_hist:
		num_photos_hist[len(biz_photo_dict[key])] = 0
	num_photos_hist[len(biz_photo_dict[key])] += 1

print 'average #photos per biz: ', photo_total / biz_total
print 'max #photos for one biz: ', max_photo
print 'min #photos for one biz: ', min_photo
# for key in num_photos_hist:
# 	print key, num_photos_hist[key]