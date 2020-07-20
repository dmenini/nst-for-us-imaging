import os
from PIL import Image

parent = 'img/'
data_path = 'img/data/new_att_all/'

TEST_SIZE = 600

def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

make_dir(os.path.join(parent, 'lq_dataset'))
make_dir(os.path.join(parent, 'lq_test'))
make_dir(os.path.join(parent, 'hq_dataset'))
make_dir(os.path.join(parent, 'hq_test'))
make_dir(os.path.join(parent, 'seg_dataset'))
make_dir(os.path.join(parent, 'seg_test'))

for i in range(1,6669):
	filename = str(i) + '.png'

	image = Image.open(data_path + filename)

	if i <= TEST_SIZE:
		lq = image.crop((0, 0, 1386, 1000)).save(os.path.join(parent, 'lq_test', filename))
		hq = image.crop((1386, 0, 1386*2, 1000)).save(os.path.join(parent, 'hq_test', filename))
		seg = image.crop((1386*2, 0, 1386*3, 1000)).save(os.path.join(parent, 'seg_test', filename))
	else:
		lq = image.crop((0, 0, 1386, 1000)).save(os.path.join(parent, 'lq_dataset', filename))
		hq = image.crop((1386, 0, 1386*2, 1000)).save(os.path.join(parent, 'hq_dataset', filename))
		seg = image.crop((1386*2, 0, 1386*3, 1000)).save(os.path.join(parent, 'seg_dataset', filename))		
