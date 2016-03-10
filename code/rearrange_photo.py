import os, sys

mapping = sys.argv[1]
data_dir = sys.argv[2]

with open('%s' % mapping, 'r') as f:
    for line in f:
        photo, business = line[:-1].split(',')
        if not os.path.isfile('%s%s.jpg' % (data_dir, photo)):
            continue
        path = '%s%s' % (data_dir, business)
        if not os.path.exists(path):
            os.mkdir(path)
        os.rename('%s%s.jpg' % (data_dir, photo), '%s/%s.jpg' % (path, photo))
