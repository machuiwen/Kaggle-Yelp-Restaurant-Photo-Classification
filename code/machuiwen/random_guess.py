import random

fin = open("sample_submission.csv", 'r')
fout = open("random_submission.csv", 'w')
fout.write(fin.readline())

for line in fin:
	bid = line.split(',')[0]
	labels = ""
	for i in range(9):
		if random.random() < 0.5:
			labels += str(i) + ' '
	fout.write(bid + ',' + labels.strip() + '\n')

fin.close()
fout.close()