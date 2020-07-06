import numpy as np
import pickle
import tensorflow as tf

import utils

from PIL import Image

clinical = 0
N_IMG = 600

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']

def main():

	# load vgg network
	extractor = utils.StyleContentModel(style_layers, content_layers)

	style_features_avg = [0.0, 0.0, 0.0, 0.0, 0.0]

	for i in range(1,N_IMG):

		if clinical:
			image_path = 'img/clinical_us/training_set/' + format(i, '03d') + '_HC.png'
			style_image = utils.image_preprocessing(image_path, 'clinical', [540, 800], c=3)
		else:
			image_path = 'img/data/new_att_all/' + str(i) + '.png'
			style_image = utils.image_preprocessing(image_path, 'hq', [1000, 1386], c=3)

		print(image_path)

		# calculate gram matrices for style feature layer maps we care about
		style_features = extractor(style_image)['style']
		style_features_list = [style_features[name]  for name in style_features.keys()]

		# style_gram = [gram_matrix(style_features[name]) for name in style_features.keys()]

		for j in range(5):
			style_features_avg[j] += style_features_list[j]
			# style_grams[j] += style_gram[j]

	style_features_avg = [ft/(N_IMG - 1) for ft in style_features_avg]

	style_dict = {name: value for name, value in zip(style_layers, style_features_avg)}

	if clinical:
		filename = 'models/nst/us_clinical_ft_dict.pickle'
	else:
		filename = 'models/nst/us_hq_ft_dict.pickle'
	with open(filename, 'wb') as handle:
		pickle.dump(style_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gram_matrix(input_tensor):
	""" 
	Calculate style by means of Gram Matrix, which include means and correlation across feature maps.
	"""
	result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
	return result / (num_locations)


if __name__ == "__main__":
	main()