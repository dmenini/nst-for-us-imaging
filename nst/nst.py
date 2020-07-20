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
import pickle

from PIL import Image
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0=all', '1=I', '2=IW' '3=IWE'}

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

parser = argparse.ArgumentParser(description='Optimization parameters and files.')
# Passed by script eval_*.sh
parser.add_argument('--data-dir', dest='data_dir',    type=str,     default='img/data/new_att_all',     		help='Directory containing input images')
parser.add_argument('--save-dir', dest='save_dir',    type=str,     default='output/result',            		help='Directory where to store the results')
parser.add_argument('--content',  dest='content',     type=str,     default='img/data/new_att_all/34.png',  	help='Content image path')
parser.add_argument('--style', 	  dest='style',       type=str,     default='img/data/new_att_all/665.png',  	help='Style image path')
parser.add_argument('--dict-path',dest='dict_path',   type=str,     default='models/nst/us_hq_ft_dict.pickle',  help='Path to the dict to be used as style target')
parser.add_argument('--task', 	  dest='task',        type=str,     default='lq2hq',            				help='Task')
parser.add_argument('--name', 	  dest='name',        type=str,     default='',            						help='Prefix of the output image')
parser.add_argument('--loss',     dest='loss',  	  type=int,     default=0,     								help='0=StyleLoss, 1=StyleLoss+, 2=AverageStyle')
# Passed by user (or default)
parser.add_argument('--lr',       dest='lr',  	  	  type=float,   default=0.005,     							help='Learning rate')
parser.add_argument('--weights',  dest='weights',     type=float,   default=[1e-2, 1e5],   	 nargs='+', 		help='Style and content weights')
parser.add_argument('--epochs',   dest='epochs',      type=int,     default=300,                        		help='Max number of epochs')
parser.add_argument('--steps',    dest='steps',       type=int,     default=10,                         		help='Number of steps per epoch')
parser.add_argument('--size',     dest='input_size',  type=int,     default=1386,                       		help='input size (max dim)')
parser.add_argument('--message',  dest='message',	  type=str,		default='',									help='Submission description')

# Style layer 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
# Content layer 
content_layers = ['block5_conv2']

# Global variables
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

	labels = [int(value) for value in labels if 100 <= value <= 220]
	masks = [image_to_tensor((np.array(seg_image) == label).astype(np.float32), c=c) for label in labels]
	mask_dict = {label: mask for mask, label in zip(masks, labels)}

	mask0 = image_to_tensor((np.array(seg_image) > 59).astype(np.float32), c=c)
	mask1 = image_to_tensor((np.array(seg_image) > 218).astype(np.float32), c=c)
	mask_dict.update({'obj_full': (mask0 - mask1)})
	mask0 = image_to_tensor((np.array(seg_image) > 200).astype(np.float32), c=c)
	mask1 = image_to_tensor((np.array(seg_image) > 220).astype(np.float32), c=c)
	mask_dict.update({'obj': (mask0 - mask1)})
	mask_fg = image_to_tensor((np.array(seg_image) >= 18).astype(np.float32), c=c)
	mask_dict.update({'fg': mask_fg})

	if visualize:
		for l in mask_dict.keys():
			tensor_to_image(tf.image.grayscale_to_rgb(mask_dict[l])).save('output/masks/' + str(l) + '.png')

	mask_dict = {label: mask_dict[label] for label in mask_dict.keys() if np.sum(mask_dict[label]) > w*h*c*0.0025}     # Filter out small masks
	mask_dict = {label: tf.image.resize(mask_dict[label], style_shape, method='nearest') for label in mask_dict.keys()}

	print("Segmentation labels: ", mask_dict.keys())

	return mask_dict


def truncate_masks(style_masks, content_masks):
	"""
	Truncate style and content masks to common labels only.
	"""
	common_labels = [label for label in content_masks.keys() if label in style_masks.keys()]
	print("Common labels: ", common_labels)
	style_masks = 	[style_masks[label]	for label in style_masks.keys() if label in common_labels]
	content_masks = [content_masks[label] for label in content_masks.keys() if label in common_labels]

	return style_masks, content_masks


def gram_matrix(input_tensor):
	result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
	return result / (num_locations)


def style_augmented_loss(outputs, masks_content, masks_style, style_features, style_weight):
	"""
	Style loss augmented with semantic information from style and content masks.
	"""
	style_outputs = outputs['style']
	num_style_layers = len(style_outputs)
	
	layer_style_loss = [0.0] * num_style_layers

	for mask_content, mask_style in zip(masks_content, masks_style):
		layer_mask_content = resize_mask(mask_content, style_layers)
		layer_mask_style = resize_mask(mask_style, style_layers)
		for i, name in enumerate(style_outputs.keys()):
			gm_outputs = gram_matrix(tf.multiply(style_outputs[name], layer_mask_content[name]))
			gm_features = gram_matrix(tf.multiply(style_features[name], layer_mask_style[name]))

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


def scores(stylized_image, style_target):
	mse_score = mse(stylized_image, style_target)
	psnr_score = tf.image.psnr(stylized_image, style_target, max_val=1.0).numpy()[0]
	ssim_score = tf.image.ssim(stylized_image, style_target, max_val=1.0).numpy()[0]
	print("\tSCORE:\tMSE = {:.6f} \tPSNR = {:.6f} \tSSIM = {:.6f}".format(mse_score, psnr_score, ssim_score))


def neural_style_transfer(args):

	content_path = args.content
	style_path = args.style

	if args.loss == 0:
		print("Neural style transfer with basic style loss. \nContent image = {}\nStyle image = {}".format(content_path, style_path))
	elif args.loss == 1:
		print("Neural style transfer with augmented style loss. \nContent image = {}\nStyle image = {}".format(content_path, style_path))
	elif args.loss == 2:
		print("Neural style transfer with average style loss. \nContent image = {}\nStyle image = {}".format(content_path, style_path))
	else:
		print("Wrong loss")
		return 0		

	if args.task == 'lq2hq':
		content = image_preprocessing(content_path, 'lq', [round(args.input_size/1.386), args.input_size], c=3)
		content_seg = image_preprocessing(content_path, 'segmentation', [round(args.input_size/1.386), args.input_size], c=3)	
		style_target = image_preprocessing(content_path, 'hq', [round(args.input_size/1.386), args.input_size], c=3)
		style = image_preprocessing(style_path, 'hq', [round(args.input_size/1.386), args.input_size], c=3)
		style_seg = image_preprocessing(style_path, 'segmentation', [round(args.input_size/1.386), args.input_size], c=3)

	elif args.task == 'seg2hq':
		content = image_preprocessing(content_path, 'segmentation', [round(args.input_size/1.386), args.input_size], c=3)
		content_seg = content
		style_target = image_preprocessing(content_path, 'hq', [round(args.input_size/1.386), args.input_size], c=3)
		style = image_preprocessing(style_path, 'hq', [round(args.input_size/1.386), args.input_size], c=3)
		style_seg = image_preprocessing(style_path, 'segmentation', [round(args.input_size/1.386), args.input_size], c=3)

	elif args.task == 'hq2clinical':
		content = image_preprocessing(content_path, 'hq', [round(args.input_size/1.386), args.input_size], c=3)
		content_seg = image_preprocessing(content_path, 'segmentation', [round(args.input_size/1.386), args.input_size], c=3)	
		style = image_preprocessing(style_path, 'clinical', [round(args.input_size/1.386), args.input_size], c=3)
		style_seg = content_seg			# Not correct eventually, but works...

	else:
		print("Wrong task")
		return 0

	# ==================================================================================================================
	# Extract style features, content features and masks
	# ==================================================================================================================

	mask_dict_content = extract_masks(content_seg, [style.shape[1], style.shape[2]], c=1, visualize=False)
	print("Number of content masks: {}".format(len(mask_dict_content)))

	mask_dict_style = extract_masks(style_seg, [style.shape[1], style.shape[2]], c=1, visualize=False)
	print("Number of style masks: {}".format(len(mask_dict_style)))

	style_masks, content_masks = truncate_masks(mask_dict_style, mask_dict_content)

	extractor = StyleContentModel(style_layers, content_layers)
	style_features = extractor(style)['style']
	content_features = extractor(content)['content']

	style_grams = {}
	if args.loss == 2:
		print('Loading dict from {}'.format(args.dict_path))
		with open(args.dict_path, 'rb') as handle:
			style_features = pickle.load(handle)

	# ==================================================================================================================
	# Run gradient descent
	# ==================================================================================================================

	init_image = np.random.randn(1, content.shape[1], content.shape[2], content.shape[3]).astype(np.float32) * 0.0001
	stylized_image = tf.Variable(init_image)

	opt = tf.optimizers.Adam(learning_rate=args.lr, beta_1=0.99, epsilon=1e-7)

	@tf.function()
	def train_step(image, masks_content, masks_style, style_features, content_features, weights):
		image.assign(tf.multiply(image, mask_dict_content['fg']))
		with tf.GradientTape() as tape:
			outputs = extractor(image)
			if args.loss == 0 or args.loss == 2:
				loss_style = style_loss(outputs, style_features, weights[0])
			elif args.loss == 1:
				loss_style = style_augmented_loss(outputs, masks_content, masks_style, style_features, weights[0])
			loss_content = content_loss(outputs, content_features, weights[1])
			loss = loss_style + loss_content
		grad = tape.gradient(loss, image)
		opt.apply_gradients([(grad, image)])
		image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
		return loss, loss_style, loss_content

	best_loss = np.float("inf")
	best_image = None
	start = time.time()

	for n in range(args.epochs):
		print("Epoch: {}".format(n))
		for m in range(args.steps):
			loss, loss_style, loss_content = train_step(stylized_image, content_masks, style_masks, style_features, content_features, args.weights)
		print("\tLOSS:\tstyle = {:.2f} \tcontent = {:.2f} \tTOT = {:.2f}".format(loss_style, loss_content, loss))
		if not args.task == 'hq2clinical':
			scores(stylized_image, style_target)
		file_name = args.save_dir + '/opt/ep_' + str(n) + '.png'
		if n % SAVE_INTERVAL == 0:
			tensor_to_image(stylized_image).convert('L').save(file_name)
		if loss < best_loss:
			best_loss = loss
			best_image = stylized_image
			best_epoch = n

	end = time.time()
	print("Total time: {:.1f}\n".format(end - start))

	best_image = tf.multiply(best_image, mask_dict_content['fg'])
	if not args.task == 'hq2clinical':
		print("FINAL SCORES (epoch {}):".format(best_epoch))
		scores(best_image, style_target)
	print("\n")
	image_number = [int(s) for s in re.split('[/ .]',content_path) if s.isdigit()][0]
	file_name = args.save_dir + '/' + str(args.task) + '_' + str(args.name) + str(image_number) + '.png'
	tensor_to_image(best_image).convert('L').save(file_name)


def main():

	args = parser.parse_args()

	print(args)
	print(args.message)

	neural_style_transfer(args)

	return 0


if __name__ == "__main__":
	main()
