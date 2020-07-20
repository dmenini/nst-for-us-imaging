import numpy as np
import tensorflow as tf
import pathlib
import os
from argparse import ArgumentParser
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from imageio import imread
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0=all', '1=I', '2=IW' '3=IWE'}

parser = ArgumentParser()
parser.add_argument("path", type=str, nargs=2, help='Path to the generated images')

# scale an array of images to a new size
def crop_images(images, new_height, new_width, pos):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		image = Image.fromarray(image).convert('RGB')
		width, height = image.size   # Get dimensions
		if pos == 'top_left':
			left = width//2 - new_width
			top = height//2 - new_height
		elif pos == 'top_right':
			left = width//2
			top = height//2 - new_height
		elif pos == 'bottom_left':
			left = width//2 - new_width
			top = height//2
		elif pos == 'bottom_right':
			left = width//2
			top = height//2
		elif pos == 'center':
			left = (width - new_width)//2
			top = (height - new_height)//2
		right = (left + new_width)
		bottom = (top + new_height)					
		# Crop the center of the image
		new_image = image.crop((left, top, right, bottom))
		# store
		images_list.append(np.array(new_image).astype(np.float32))
	return np.asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def main():

	args = parser.parse_args()
	print(args)
	
	# prepare the inception v3 model
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

	pos = ['top_right', 'top_left', 'bottom_left', 'bottom_right']
	# pos = ['center']
	fid = np.array([0] * len(pos))

	for i, p in enumerate(pos):
		path1 = pathlib.Path(args.path[0])
		files1 = list(path1.glob('*.png'))
		images1 = np.array([imread(str(fn)) for fn in files1])
		images1 = crop_images(images1, 299, 299, p)
		images1 = preprocess_input(images1)

		path2 = pathlib.Path(args.path[1])
		files2 = list(path2.glob('*.png'))
		images2 = np.array([imread(str(fn)) for fn in files2])
		images2 = crop_images(images2, 299, 299, p)
		images2 = preprocess_input(images2)

		# fid between images1 and images2
		fid[i] = calculate_fid(model, images1, images2)

	fid = np.mean(fid)		
	print('FID: %.3f' % fid)

	return 0


if __name__ == "__main__":
	main()