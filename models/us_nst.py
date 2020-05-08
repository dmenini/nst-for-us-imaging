from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from nst_lib import *
from img_lib import *

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

parser = argparse.ArgumentParser(description='Optimization parameters and files.')
parser.add_argument('--data-dir', metavar='data_dir', type=str,
                    default='img/data/new_att_all/', help='Directory containing inputs.')
parser.add_argument('--save-dir', metavar='save_dir', type=str,
                    default='img/result/', help='Directory containing inputs.')
parser.add_argument('--image', metavar='image', type=int, nargs='+',
                    default=[1, 18, 34], help='Image number.')
parser.add_argument('--weights', metavar='weights', type=float, nargs='+',
                    default=[1e2, 1], help='Style and content weights.')
parser.add_argument('--epochs', metavar='weights', type=int,
                    default=50, help='Max number of epochs.')
parser.add_argument('--steps', metavar='steps_per_epoch', type=int,
                    default=15, help='Number of steps per epoch.')
parser.add_argument('--size', metavar='input_size', type=int,
                    default=1386, help='Number of steps per epoch.')
args = parser.parse_args()

style_weight = args.weights[0]
content_weight = args.weights[1]
epochs = args.epochs
steps_per_epoch = args.steps
input_size = args.size

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def main():
    print(args)

    for i in args.image:
        image_path = args.data_dir + str(i) + '.png'
        print(image_path)
        content_image = image_preprocessing(image_path, 'content', input_size, c=3)
        style_image = image_preprocessing(image_path, 'style', input_size, c=3)

        # plt.subplot(1, 3, 1)
        # imgshow(content_image, title='Content Image (' + str(content) + ')')
        # plt.subplot(1, 3, 2)
        # imgshow(style_image, title='Style Image (HQ)')

        # stylized_image = quick_nst(content_image, style_image)
        stylized_image, score = long_nst(content_image, style_image, reg=True)

        # plt.subplot(1, 3, 3)
        # imgshow(rgb2gray(stylized_image), title='Stylized Image')

        # plt.show(block=False)
        # plt.pause(2)
        # plt.close()

        file_name = args.save_dir + 'basic_' + str(i) + '_' + str(int(np.round(score))) + '.png'
        tensor_to_image(stylized_image).convert('L').save(file_name)


def quick_nst(content_image, style_image):

    nst_module = tf.saved_model.load("./nst_model")

    stylized_image = nst_module(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image


def long_nst(content_image, style_image, reg=True):

    # ==================================================================================================================
    # Create a model that extracts both content and style
    # ==================================================================================================================

    # This model returns a dict of the gram matrix (style) of the style_layers and content of the content_layers
    extractor = StyleContentModel(style_layers, content_layers)

    # ==================================================================================================================
    # Run gradient descent (with regularization term in the loss function)
    # ==================================================================================================================

    # Set style and content target values
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # Define a tf.Variable to contain the image to optimize
    stylized_image = tf.Variable(content_image)
    best_image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs):
        """Weighted combination of style and content loss"""
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        #image.assign(tf.repeat(tf.image.rgb_to_grayscale(image), 3, -1))

    start = time.time()
    score_max = 0

    step = 0
    for n in range(epochs):
        print("Epoch: {}".format(n))
        for m in range(steps_per_epoch):
            train_step(stylized_image)
        mse_score = mse(pil_grayscale(stylized_image), pil_grayscale(style_image))    
        psnr_score = psnr(pil_grayscale(stylized_image), pil_grayscale(style_image))
        ssim_score = tf.image.ssim(stylized_image, style_image, max_val=1.0).numpy()[0]
        print("\tMSE = {} \tPSNR = {} \tSSIM = {}".format(mse_score, psnr_score, ssim_score))
        file_name = args.save_dir + 'opt/ep_' + str(n) + '.png'
        tensor_to_image(stylized_image).save(file_name)
        if ssim_score > score_max:
            score_max = ssim_score
            best_image = stylized_image

    end = time.time()
    print("Total time: {:.1f}\n".format(end - start))

    return best_image, score_max


if __name__ == "__main__":
    main()
