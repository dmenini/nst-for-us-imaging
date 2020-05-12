import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import pytorch_ssim
import utils
from network import ImageTransformNet
from vgg16 import Vgg16

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='style transfer in pytorch')
parser.add_argument('--mode',       dest='mode',        type=str,   default='train',                help='Train or transfer')
parser.add_argument('--data-dir',   dest='data_dir',    type=str,   default='img/data/new_att_all', help='Directory containing input images.')
parser.add_argument('--save-dir',   dest='save_dir',    type=str,   default='img/result',           help='Directory where to store the results.')
parser.add_argument('--image',      dest='image',       type=int,   default=1,                      help='Input image number.')
parser.add_argument("--model-path", dest='model_path',  type=str,   default='perc_loss',            help="Path to write/read a pretrained model for a style image")
parser.add_argument('--gpu',        dest='gpu',         type=int,   default=0,                      help='Use GPU or not.')
parser.add_argument('--style-dir',  dest='style_dir',   type=str,   default='img/style_dataset',    help='Path to directory of style_loss.')
parser.add_argument("--visualize",  dest='visualize',   type=int,   default=1,                      help="Set to 1 if you want to visualize training")
parser.add_argument("--weights",    dest='weights',     type=float, default=[1e5, 1e0, 1e-7],       help="weight of style loss, content loss, tv loss", nargs='+')
parser.add_argument("--batch-size", dest='batch_size',  type=int,   default=4,                      help="batch size")
parser.add_argument("--image-size", dest='image_size',  type=int,   default=512,                    help="input size")
parser.add_argument("--epochs",     dest='epochs',      type=int,   default=10,                     help="epochs")


def train(args):

    dtype = torch.float64 
    if args.gpu:
        use_cuda = True
        print("Current device: %d" %torch.cuda.current_device())
        dtype = torch.cuda.FloatTensor

    # visualization of training controlled by flag
    if (args.visualize):
        img_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(args.image_size),                  # scale shortest side to image_size
            transforms.CenterCrop(args.image_size),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])
        image_path = args.data_dir + '/' + str(args.image) + '.png'
        testImage = utils.load_image(image_path)
        w, h = testImage.size
        testImage = testImage.crop((0, 0, w/3, h))
        testImage = img_transform(testImage)
        testImage = Variable(testImage.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # define network
    image_transformer = ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), 1e-3) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().type(dtype)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(args.image_size),          # scale shortest side to image_size
        transforms.CenterCrop(args.image_size),      # crop center image_size out
        transforms.ToTensor(),                       # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()           # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.style_dir, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),             # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    image_path = args.style_dir + '/new_att_all/' + str(args.image) + '.png'
    style = utils.load_image(image_path)
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
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = args.weights[0]*style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
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
                status = "Epoch {}:\t [{}/{}]\t Batch:[{}]\t agg_style: {:.6f}\t agg_content: {:.6f}\t agg_tv: {:.6f}\t style: {:.6f}\t content: {:.6f}\t tv: {:.6f}".format(
                                e+1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item()
                            )
                print(status)

        if (args.visualize):
            image_transformer.eval()

            outputTestImage = image_transformer(testImage).cpu()
            out_path = args.save_dir + "/opt/perc%d_%d.png" %(args.image, e+1)
            utils.save_image(out_path, outputTestImage.data[0])

            image_transformer.train()

    # save model
    image_transformer.eval()

    filename = args.model_path + '/us_style_' + str(args.image_size) + '.model'
    torch.save(image_transformer.state_dict(), filename)


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
            transforms.Resize(args.image_size),                 # scale shortest side to image_size
            transforms.CenterCrop(args.image_size),             # crop center image_size out
            transforms.ToTensor(),                              # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()                  # normalize with ImageNet values
    ])

    image_path = args.data_dir + '/' + str(args.image) + '.png'
    image = utils.load_image(image_path)
    w, h = image.size

    style = image.crop((w/3, 0, w/3*2, h))
    style = img_transform(style)
    style = style.unsqueeze(0)

    content = image.crop((0, 0, w/3, h))
    content = img_transform(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    filename = args.model_path + '/us_style_512.model'
    style_model = ImageTransformNet().type(dtype)
    if args.gpu == 0:
        style_model.load_state_dict(torch.load(filename, map_location='cpu'))
    else:
        style_model.load_state_dict(torch.load(filename))

    n_par = sum(p.numel() for p in style_model.parameters() if p.requires_grad)
    print("The pretrained model {} has {} parameters.".format(filename, n_par))

    # process input image
    stylized = style_model(content).cpu()

    stylized = utils.save_image(args.save_dir + '/perc' + str(args.image) + '.png', stylized.data[0])
    style = utils.save_image(args.save_dir + '/style' + str(args.image) + '.png', style.data[0])

    stylized = torch.tensor(np.array(stylized.convert('RGB'))).unsqueeze(0)
    style = torch.tensor(np.array(style.convert('RGB'))).unsqueeze(0)

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
        print("Style transfering!")
        style_transfer(args)
    else:
        print("invalid command")


if __name__ == '__main__':
    main()