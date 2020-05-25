import numpy as np
import torch
import os
import argparse
import time
import math

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models

import pytorch_ssim
import utils
from network import ImageTransformNet
from vgg19 import Vgg19

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='style transfer in pytorch')
parser.add_argument('--mode',       dest='mode',        type=str,   default='train',                help='Train or transfer')
parser.add_argument('--data-dir',   dest='data_dir',    type=str,   default='img/coco_val/val2017', help='Directory containing input images.')
parser.add_argument('--save-dir',   dest='save_dir',    type=str,   default='output/result',        help='Directory where to store the results.')
parser.add_argument('--dataset',    dest='dataset',     type=str,   default='img/coco_val',         help='Path to directory containing the dataset.')
parser.add_argument("--model-name", dest='model_name',  type=str,   default='us_style',             help="Name of the pretrained model to write/read")
parser.add_argument('--image',      dest='image',       type=int,   default=1,                      help='Input image number (1-34).')
parser.add_argument('--gpu',        dest='gpu',         type=int,   default=0,                      help='Use GPU or not.')
parser.add_argument("--weights",    dest='weights',     type=float, default=[1e5, 1e0, 1e-4],       help="weight of style loss, content loss, tv loss", nargs='+')
parser.add_argument("--batch-size", dest='batch_size',  type=int,   default=4,                      help="batch size")
parser.add_argument("--image-size", dest='image_size',  type=int,   default=512,                    help="input size (min dim)")
parser.add_argument("--epochs",     dest='epochs',      type=int,   default=5,                      help="epochs")

# Global variables
norm = True
SHAPE_RATIO = 1.386


def train(args):

    dtype = torch.float64 
    if args.gpu:
        use_cuda = True
        print("Current device: %d" %torch.cuda.current_device())
        dtype = torch.cuda.FloatTensor

    #image_path = args.data_dir + '/' + str(args.image) + '.png'     # Image from 1 to 34 
    image_path = args.data_dir + "/000000000285.jpg"
    image = utils.load_image(image_path)
    w, h = image.size

    # visualization of training
    img_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(args.image_size, interpolation=Image.NEAREST),    # scale shortest side to image_size
        transforms.CenterCrop(args.image_size),                             # crop center image_size out
        transforms.ToTensor(),                                              # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform(norm)                              # normalize with ImageNet values
    ])
    
    #testImage = image.crop((0, 0, w/3, h))
    testImage = img_transform(image)
    testImage = Variable(testImage.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # define network
    image_transformer = ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), 1e-3) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg19().type(dtype)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(args.image_size, interpolation=Image.NEAREST),            # scale shortest side to image_size
        transforms.CenterCrop(args.image_size),                                     # crop center image_size out
        transforms.ToTensor(),                                                      # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform(norm)                                      # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),                      # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform(norm)      # normalize with ImageNet values
    ])

    #style = image.crop((round(w/3), 0, round(w/3*2), h))
    style = utils.load_image('/scratch_net/hoss/dmenini/nst-for-us-imaging/img/Vassily_Kandinsky,_1913_-_Composition_7.jpg')
    style = style_transform(style)
    style = Variable(style.repeat(args.batch_size, 1, 1, 1)).type(dtype)

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    for e in range(args.epochs):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

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
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(5):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = args.weights[0]*style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (block5_conv2)
            recon = y_c_features[5]      
            recon_hat = y_hat_features[5]
            content_loss = args.weights[1]*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization (anisotropic version)
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = args.weights[2]*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = "Epoch {}:\t [{}/{}]\t\t Batch:[{}]\t agg_style: {:.6f}\t agg_content: {:.6f}\t agg_tv: {:.6f}\t style: {:.6f}\t content: {:.6f}\t tv: {:.6f}".format(
                                e, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item()
                            )
                print(status)

        image_transformer.eval()

        outputTestImage = image_transformer(testImage).cpu()
        out_path = args.save_dir + "/opt/perc%d_%d_%d.png" %(args.image, e, batch_num+1)
        utils.save_image(out_path, outputTestImage.data[0], norm)

        image_transformer.train()

    # save model
    image_transformer.eval()

    filename = 'models/perceptual/' + str(args.model_name) + '.model'
    torch.save(image_transformer.state_dict(), filename)


def style_transfer(args):
    # works on cpu, not sure on gpu

    W = math.ceil(args.image_size*SHAPE_RATIO) - 2

    dtype = torch.float64 
    if args.gpu:
        use_cuda = True
        print("Current device: %d" %torch.cuda.current_device())
        dtype = torch.cuda.FloatTensor

    # content image
    img_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(args.image_size, interpolation=Image.NEAREST),                 # scale shortest side to image_size
            transforms.CenterCrop((args.image_size,W)),       # crop center image_size out
            transforms.ToTensor(),                              # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform(norm)              # normalize with ImageNet values
    ])

    image_path = args.data_dir + '/' + str(args.image) + '.png'
    image = utils.load_image(image_path)
    w, h = image.size

    style = image.crop((w/3, 0, w/3*2, h))
    style = img_transform(style)
    style = style.unsqueeze(0)

    content = image.crop((0, 0, w/3, h))
    # content = image.crop((w/3*2, 0, w, h)) # segmentation image
    content = img_transform(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    filename = 'models/perceptual/' + str(args.model_name) + '.model'
    style_model = ImageTransformNet().type(dtype)
    if args.gpu == 0:
        style_model.load_state_dict(torch.load(filename, map_location='cpu'))
    else:
        style_model.load_state_dict(torch.load(filename))

    n_par = sum(p.numel() for p in style_model.parameters() if p.requires_grad)
    print("The pretrained model {} has {} parameters.".format(filename, n_par))

    # process input image
    stylized = style_model(content).cpu()

    utils.save_image(args.save_dir + '/perc' + str(args.image) + '.png', stylized.data[0], norm)

    stylized = torch.tensor(utils.denormalize_tensor_transform(stylized.data[0], norm)).unsqueeze(0)
    style = torch.tensor(utils.denormalize_tensor_transform(style.data[0], norm)).unsqueeze(0)

    mse_score = torch.mean((stylized * 1.0 - style * 1.0) ** 2)
    psnr_score = 20 * torch.log10(255.0 / torch.sqrt(mse_score))
    ssim_score = pytorch_ssim.ssim(stylized.type(torch.DoubleTensor), style.type(torch.DoubleTensor)).item()
    print("\tSCORE:\tMSE = {:.6f} \tPSNR = {:.6f} \tSSIM = {:.6f}".format(mse_score, psnr_score, ssim_score))


def main():
    args = parser.parse_args()
    print(args)

    if args.mode == "train":
        print("Training!")
        train(args)
    elif args.mode == "transfer":
        print("Style transfer!")
        style_transfer(args)
    else:
        print("invalid command")


if __name__ == '__main__':
    main()