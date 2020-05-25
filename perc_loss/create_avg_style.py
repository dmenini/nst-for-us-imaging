import numpy as np
import torch
import pickle

import utils
from vgg19 import Vgg19
from torchvision import transforms

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dtype = torch.float64 
norm = True

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# load vgg network
vgg = Vgg19().type(dtype)
style_grams = [0.0, 0.0, 0.0, 0.0, 0.0]
MAX_IMG = 1000

for i in range(1,MAX_IMG):

	image_path = 'img/style_dataset/new_att_all/' + str(i) + '.png'
	image = utils.load_image(image_path)
	print(image_path)

	# style image
	style_transform = transforms.Compose([
		transforms.ToTensor(),                      # turn image from [0-255] to [0-1]
		utils.normalize_tensor_transform(norm)      # normalize with ImageNet values
	])

	style = style_transform(image)
	style = style.repeat(1, 1, 1, 1).type(dtype)

	# calculate gram matrices for style feature layer maps we care about
	style_features = vgg(style)
	style_gram = [utils.gram(fmap) for fmap in style_features]

	for j in range(5):
		style_grams[j] += style_gram[j]

style_grams = [gram / (MAX_IMG - 1) for gram in style_grams]

for j in range(5):
	style_dict = {name: value for name, value in zip(style_layers, style_grams)}

with open('models/perceptual/us_style_dict.pickle', 'wb') as handle:
    pickle.dump(style_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

