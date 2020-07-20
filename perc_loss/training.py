import numpy as np
import torch
import os
import re
import argparse
import time
import math
import matplotlib
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
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
parser.add_argument('--save-dir',   dest='save_dir',    type=str,   default='output/result',                    	help='Directory where to store the results.')
parser.add_argument('--dataset',    dest='dataset',     type=str,   default='img/lq_dataset/',             			help='Path to directory containing the dataset.')
parser.add_argument("--model-name", dest='model_name',  type=str,   default='us_lq2hq',                         	help="Name of the pretrained model to write/read")
parser.add_argument('--content',    dest='content',     type=str,   default='img/lq_test/34.png',                   help='Content image (test).')
parser.add_argument('--style',      dest='style',       type=str,   default='img/hq_dataset/new_att_all/645.png',   help='Style image.')
parser.add_argument('--gpu',        dest='gpu',         type=int,   default=0,                                  	help='Use GPU or not.')
parser.add_argument("--weights",    dest='weights',     type=float, default=[5e6, 1e2],                   		    help="weight of style loss, content loss", nargs='+')
parser.add_argument("--batch-size", dest='batch_size',  type=int,   default=1,                                  	help="batch size")
parser.add_argument("--image-size", dest='image_size',  type=int,   default=1000,                                	help="input size (min dim)")
parser.add_argument("--epochs",     dest='epochs',      type=int,   default=5,                                  	help="epochs")
parser.add_argument('--loss',       dest='loss',        type=int,   default=0,                                      help='0=StyleLoss, 1=AverageStyle')

# Global variables
norm = True

def training(args):

    dtype = torch.float64 
    if args.gpu:
        use_cuda = True
        print("Current device: %d" %torch.cuda.current_device())
        dtype = torch.cuda.FloatTensor

    print('content = {}'.format(args.content))
    print('style = {}'.format(args.style))

    img_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(args.image_size, interpolation=Image.NEAREST),    # scale shortest side to image_size
        transforms.CenterCrop(args.image_size),                             # crop center image_size out
        transforms.ToTensor(),                                              # turn image from [0-255] to [0-1]
        utils.normalize_imagenet(norm)                                      # normalize with ImageNet values
    ])

    content = Image.open(args.content)
    content = img_transform(content)        # Loaded already cropped
    content = Variable(content.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # define network
    image_transformer = ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), 1e-5) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg19().type(dtype)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(args.image_size, interpolation=Image.NEAREST),    # scale shortest side to image_size
        transforms.CenterCrop(args.image_size),                             # crop center image_size out
        transforms.ToTensor(),                                          # turn image from [0-255] to [0-1]
        utils.normalize_imagenet(norm)                                  # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)

    # style image
    style_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),                          # turn image from [0-255] to [0-1]
        utils.normalize_imagenet(norm)                  # normalize with ImageNet values
    ])

    style = Image.open(args.style)
    if "clinical" in args.style:
        style = style.crop((20,0,style.size[0],style.size[1]))      # Remove left bar from the style image
    style = style_transform(style)
    style = Variable(style.repeat(args.batch_size, 1, 1, 1)).type(dtype)

    # calculate gram matrices for target style layers
    style_features = vgg(style)
    style_gram = [utils.gram(feature) for feature in style_features]

    if args.loss == 1:
        print("Using average style on features")
        if "clinical" in args.style:
            with open('models/perceptual/us_clinical_ft_dict.pickle', 'rb') as handle:
                style_features = pickle.load(handle) 
        else:  
            with open('models/perceptual/us_hq_ft_dict.pickle', 'rb') as handle:
                style_features = pickle.load(handle) 
        style_features = [style_features[label].type(dtype) for label in style_features.keys()]
        style_gram = [utils.gram(feature) for feature in style_features]

    style_loss_list, content_loss_list, total_loss_list = [], [], []

    for e in range(args.epochs):
        count = 0
        img_count = 0

        # train network
        image_transformer.train()
        for batch_num, (x, label) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).type(dtype)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(feature) for feature in y_hat_features]
            style_loss = 0.0
            for j in range(5):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = args.weights[0] * style_loss

            # calculate content loss (block5_conv2)
            recon = y_c_features[5]      
            recon_hat = y_hat_features[5]
            content_loss = args.weights[1]*loss_mse(recon_hat, recon)

            # total loss
            total_loss = style_loss + content_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                count = count + 1
                total_loss_list.append(total_loss.item())
                content_loss_list.append(content_loss.item())
                style_loss_list.append(style_loss.item())
                print("Epoch {}:\t [{}/{}]\t\t Batch:[{}]\t total: {:.6f}\t style: {:.6f}\t content: {:.6f}\t".format(
                    e, img_count, len(train_dataset), batch_num+1, total_loss.item(), style_loss.item(), content_loss.item()))

        image_transformer.eval()

        stylized = image_transformer(content).cpu()
        out_path = args.save_dir + "/opt/perc%d_%d.png" %(e, batch_num+1)
        utils.save_image(out_path, stylized.data[0], norm)

        image_transformer.train()

    # save model
    image_transformer.eval()

    filename = 'models/perceptual/' + str(args.model_name)
    if not '.model' in filename:
        filename = filename + '.model'
    torch.save(image_transformer.state_dict(), filename)

    total_loss = np.array(total_loss_list)
    style_loss = np.array(style_loss_list)
    content_loss = np.array(content_loss_list)
    x = np.arange(0,np.size(total_loss)) / (count+1)

    fig = plt.figure('Perceptual Loss')
    plt.plot(x, total_loss)
    plt.plot(x, content_loss)
    plt.plot(x, style_loss)
    plt.legend(['Total', 'Content', 'Style'])
    plt.title('Perceptual Loss')
    plt.savefig(args.save_dir + '/perc_loss.png')


def main():
    args = parser.parse_args()
    print(args)

    training(args)


if __name__ == '__main__':
    main()