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

file = open('train.csv', 'r')
file.readline()
lst = []
for line in file:
	biz_id, labels = line.strip().split(',')
	lst.append([int(biz_id), labels])
lst = sorted(lst, key=lambda x: x[0])
for i in lst:
	print i[0],',', i[1]
	# print i[1]
