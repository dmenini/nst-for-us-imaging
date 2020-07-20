import numpy as np
import torch
import os
import re
import argparse

from torchvision import transforms
from torchvision import models

import utils
import pickle
from network import ImageTransformNet
from network import ImageTransformNet_noisy
from vgg19 import Vgg19

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='style transfer in pytorch')
parser.add_argument('--save-dir',   dest='save_dir',    type=str,   default='output/result',                        help='Directory where to store the results.')
parser.add_argument("--model-name", dest='model_name',  type=str,   default='us_lq2hq',                             help="Name of the pretrained model to read")
parser.add_argument('--content',    dest='content',     type=str,   default='img/lq_test/34.png',                   help='Content image (test).')
parser.add_argument('--style',      dest='style',       type=str,   default='img/hq_dataset/new_att_all/645.png',   help='Style image.')
parser.add_argument('--gpu',        dest='gpu',         type=int,   default=0,                                      help='Use GPU or not.')
parser.add_argument("--image-size", dest='image_size',  type=int,   default=1000,                                   help="input size (min dim)")

# Global variables
norm = True

def style_transfer(args):
    # works on cpu, not sure on gpu

    dtype = torch.float64 
    if args.gpu:
        use_cuda = True
        print("Current device: %d" %torch.cuda.current_device())
        dtype = torch.cuda.FloatTensor

    # content image
    img_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(args.image_size, interpolation=Image.NEAREST),        # scale shortest side to image_size
            transforms.CenterCrop(args.image_size),       							# crop center image_size out
            transforms.ToTensor(),                              					# turn image from [0-255] to [0-1]
            utils.normalize_imagenet(norm)              							# normalize with ImageNet values
    ])


    # load style model
    filename = 'models/perceptual/' + str(args.model_name) + '.model'
    if 'noisy' in args.model_name:
        style_model = ImageTransformNet_noisy().type(dtype)
    else:
        style_model = ImageTransformNet().type(dtype)

    if args.gpu == 0:
        style_model.load_state_dict(torch.load(filename, map_location='cpu'))
    else:
        style_model.load_state_dict(torch.load(filename))

    n_par = sum(p.numel() for p in style_model.parameters() if p.requires_grad)
    print("The pretrained model {} has {} parameters.".format(filename, n_par))

    for i in range(1,600):      # Test images
        print(i)
        content = utils.load_image(args.content + '/' + str(i) + '.png')
        content = img_transform(content)
        content = content.unsqueeze(0)
        content = Variable(content).type(dtype)

        # process input image
        stylized = style_model(content).cpu()

        utils.save_image(args.save_dir + '/' + str(args.model_name) + '/' + str(i) + '.png', stylized.data[0], norm)


def main():
    args = parser.parse_args()
    print(args)

    style_transfer(args)


if __name__ == '__main__':
    main()