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
clinical = 1

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# load vgg network
vgg = Vgg19().type(dtype)
style_features = [0.0] * len(style_layers)
MAX_IMG = 100

# style image
style_transform = transforms.Compose([
    transforms.ToTensor(),                      # turn image from [0-255] to [0-1]
    utils.normalize_imagenet(norm)      # normalize with ImageNet values
])

for i in range(0,MAX_IMG):

    if clinical:
        image_path = 'img/clinical_us/training_set/' + format(i, '03d') + '_HC.png'
    else:
        image_path = 'img/style_dataset/new_att_all/' + str(i) + '.png'

    style = Image.open(image_path).convert('RGB')
    print(image_path)
    if clinical:
        style = style.crop((20,0,style.size[0],style.size[1]))      # Remove left bar from the style image
    style = style_transform(style)
    style = style.repeat(1, 1, 1, 1).type(dtype)

    style_feature = vgg(style)
    style_feature = [fmap for fmap in style_feature]

    for j in range(len(style_layers)):
        style_features[j] += style_feature[j]

style_features = [fmap / 100 for fmap in style_features]

for j in range(len(style_layers)):
    style_dict = {name: value for name, value in zip(style_layers, style_features)}

if clinical:
    with open('models/perceptual/us_clinical_ft_dict.pickle', 'wb') as handle:
        pickle.dump(style_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('models/perceptual/us_hq_ft_dict.pickle', 'wb') as handle:
        pickle.dump(style_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)   

