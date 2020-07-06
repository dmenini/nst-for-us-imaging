from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import tensorflow as tf
import matplotlib as mpl
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import argparse

from PIL import Image
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0=all', '1=I', '2=IW' '3=IWE'}

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

parser = argparse.ArgumentParser(description='Optimization parameters and files.')
# Passed by script eval_mix.sh
parser.add_argument('--size',     dest='input_size',  type=int,     default=1386,                       	help='input size (max dim)')

# Style layer of interest
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Global variables
MAX_SIZE = 1386
SAVE_INTERVAL = 50


def resize_mask(mask, names):
	"""
	Resize the input mask as the output layers of vgg19 (style layers).
	A 3x3 convolution operation in vgg19 is an average pool on the mask (keep dimensions).
	A max pool operation in vgg19 is a bilinear resize of the mask (dimensions halved).
	"""
	def fake_pool(inputs):
		return tf.image.resize(inputs, tf.constant((int(math.floor(inputs.shape[1] / 2)), int(math.floor(inputs.shape[2] / 2)))), method='nearest')

	def fake_conv(inputs):
		return tf.nn.avg_pool(inputs, [1, 3, 3, 1], strides=None, padding='SAME')

	MaskModel = {}
	MaskModel.update({'block1_conv1': 	fake_conv(mask)})
	MaskModel.update({'block1_conv2': 	fake_conv(MaskModel['block1_conv1'])})
	MaskModel.update({'block1_pool': 	fake_pool(MaskModel['block1_conv2'])}) 

	MaskModel.update({'block2_conv1': 	fake_conv(MaskModel['block1_pool'])})
	MaskModel.update({'block2_conv2': 	fake_conv(MaskModel['block2_conv1'])})
	MaskModel.update({'block2_pool': 	fake_pool(MaskModel['block2_conv2'])})  

	MaskModel.update({'block3_conv1': 	fake_conv(MaskModel['block2_pool'])})
	MaskModel.update({'block3_conv2': 	fake_conv(MaskModel['block3_conv1'])})
	MaskModel.update({'block3_conv3': 	fake_conv(MaskModel['block3_conv2'])})
	MaskModel.update({'block3_conv4': 	fake_conv(MaskModel['block3_conv3'])})
	MaskModel.update({'block3_pool': 	fake_pool(MaskModel['block3_conv4'])}) 

	MaskModel.update({'block4_conv1': 	fake_conv(MaskModel['block3_pool'])})
	MaskModel.update({'block4_conv2': 	fake_conv(MaskModel['block4_conv1'])})
	MaskModel.update({'block4_conv3': 	fake_conv(MaskModel['block4_conv2'])})
	MaskModel.update({'block4_conv4': 	fake_conv(MaskModel['block4_conv3'])})
	MaskModel.update({'block4_pool': 	fake_pool(MaskModel['block4_conv4'])}) 

	MaskModel.update({'block5_conv1': 	fake_conv(MaskModel['block4_pool'])})
	# MaskModel.update({'block5_conv2': 	fake_conv(MaskModel['block5_conv1'])})
	# MaskModel.update({'block5_conv3': 	fake_conv(MaskModel['block5_conv2'])})
	# MaskModel.update({'block5_conv4': 	fake_conv(MaskModel['block5_conv3'])})
	# MaskModel.update({'block5_pool': 		fake_pool(MaskModel['block5_conv4'])})

	outputs = [MaskModel[name] for name in names]
	mask_dict = {name: value for name, value in zip(names, outputs)}

	return mask_dict


def extract_masks(seg_image, style_shape, c=1, visualize=False):
	"""
	From the segmentation image, extract the labels and create a binary mask for each label [18, 218].
	Additionaly, extract a mask for the main object and for the whole foreground.
	It may be needed to filter out small masks (i.e. all 0s except for some 1s) to use less memory.
	Eventually resize all the masks to the input style size.
	"""

	seg_image = tensor_to_image(seg_image).convert('L')         # Grayscale image
	w, h = seg_image.size                                  		# Get image dimensions
	pixels = list(seg_image.getdata())                    		# Get pixel list
	mask_dict = {}

	labels = [] 
	for pixel in pixels:
		if pixel not in labels:
			labels.append(pixel)
	labels = sorted(labels)

	labels = [int(value) for value in labels if 150 <= value <= 220]
	masks = [image_to_tensor((np.array(seg_image) == label).astype(np.float32), c=c) for label in labels]
	mask_dict = {label: mask for mask, label in zip(masks, labels)}

	# mask0 = image_to_tensor((np.array(seg_image) > 59).astype(np.float32), c=c)
	# mask1 = image_to_tensor((np.array(seg_image) > 218).astype(np.float32), c=c)
	# mask_dict.update({'obj_full': (mask0 - mask1)})
	# mask0 = image_to_tensor((np.array(seg_image) > 200).astype(np.float32), c=c)
	# mask1 = image_to_tensor((np.array(seg_image) > 220).astype(np.float32), c=c)
	# mask_dict.update({'obj': (mask0 - mask1)})
	# mask_fg = image_to_tensor((np.array(seg_image) >= 18).astype(np.float32), c=c)
	# mask_dict.update({'fg': mask_fg})

	if visualize:
		for l in mask_dict.keys():
			tensor_to_image(tf.image.grayscale_to_rgb(mask_dict[l])).save('output/masks/' + str(l) + '.png')

	mask_dict = {label: mask_dict[label] for label in mask_dict.keys() if np.sum(mask_dict[label]) > w*h*c*0.0025}     # Filter out small masks
	mask_dict = {label: tf.image.resize(mask_dict[label], style_shape, method='nearest') for label in mask_dict.keys()}

	print("Segmentation labels: ", mask_dict.keys())

	return mask_dict


def gram_matrix(input_tensor):
    """ 
    Calculate style by means of Gram Matrix, which include means and correlation across feature maps.
	"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def style_augmented_loss(outputs, masks, style_features, style_weight):
	""" 
    Style loss augmented with semantic information from segmentation masks
	"""
	style_outputs = outputs['style']
	num_style_layers = len(style_outputs)
	
	layer_style_loss = [0.0] * num_style_layers

	for mask in masks:
		layer_mask = resize_mask(mask, style_layers)
		for i, name in enumerate(style_outputs.keys()):
			gm_outputs = gram_matrix(tf.multiply(style_outputs[name], layer_mask[name]))
			gm_features = gram_matrix(tf.multiply(style_features[name], layer_mask[name]))

			layer_style_loss[i] += tf.reduce_mean((gm_outputs - gm_features) ** 2)

	style_loss = tf.add_n(layer_style_loss) * style_weight / num_style_layers

	return style_loss


def style_loss(outputs, style_features, style_weight):
	style_outputs = outputs['style']
	num_style_layers = len(style_outputs)
	style_loss = tf.add_n([tf.reduce_mean((gram_matrix(style_outputs[name]) - gram_matrix(style_features[name])) ** 2)
						   for name in style_outputs.keys()])
	style_loss *= style_weight / num_style_layers
	return style_loss


def content_loss(outputs, content_features, content_weight):
	content_outputs = outputs['content']
	num_content_layers = len(content_outputs)
	content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_features[name]) ** 2)
							 for name in content_outputs.keys()])
	content_loss *= content_weight / num_content_layers
	return content_loss


def total_variation_loss(output, tv_weight):
	shape = output.get_shape()
	x_deltas = tf.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
	y_deltas = tf.abs(output[:, :-1, :, :] - output[:, 1:, :, :])
	tv_loss = (tf.reduce_sum(x_deltas) + tf.reduce_sum(y_deltas)) * tv_weight
	return tv_loss

def scores(stylized_image, style_target):
	mse_score = mse(stylized_image, style_target)
	psnr_score = tf.image.psnr(stylized_image, style_target, max_val=1.0).numpy()[0]
	ssim_score = tf.image.ssim(stylized_image, style_target, max_val=1.0).numpy()[0]
	print("\tSCORE:\tMSE = {:.6f} \tPSNR = {:.6f} \tSSIM = {:.6f}".format(mse_score, psnr_score, ssim_score))


def neural_style_transfer(image, args):

	content_path = 'img/data/new_att_all/' + str(image) + '.png'

	content = image_preprocessing(content_path, 'content', [round(args.input_size/1.386), args.input_size], c=3)
	content_seg = image_preprocessing(content_path, 'segmentation', [round(args.input_size/1.386), args.input_size], c=3)	
	style = image_preprocessing(content_path, 'style', [round(args.input_size/1.386), args.input_size], c=3)
 
	# ==================================================================================================================
	# Extract style features, content features and masks
	# ==================================================================================================================

	mask_dict = extract_masks(content_seg, [style.shape[1], style.shape[2]], c=1, visualize=False)
	print("Number of content masks: {}".format(len(mask_dict)))

	masks = [mask_dict[label] for label in mask_dict.keys()]




def main():

	args = parser.parse_args()

	print(args)

	for i in range(1,35):
		neural_style_transfer(i, args)

	return 0


if __name__ == "__main__":
	main()
