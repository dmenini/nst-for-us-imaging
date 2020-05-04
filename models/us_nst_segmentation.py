from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from PIL import Image

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
                    default=[1, 18, 36], help='Image number.')
parser.add_argument('--weights', metavar='weights', type=float, nargs='+',
                    default=[1e2, 1], help='Style and content weights.')
parser.add_argument('--epochs', metavar='epochs', type=int,
                    default=25, help='Max number of epochs.')
parser.add_argument('--steps', metavar='steps_per_epoch', type=int,
                    default=50, help='Number of steps per epoch.')
parser.add_argument('--size', metavar='input_size', type=int,
                    default=1386, help='Number of steps per epoch.')
args = parser.parse_args()

style_weight = args.weights[0]
content_weight = args.weights[1]
total_variation_weight = args.weights[2]
epochs = args.epochs
steps_per_epoch = args.steps
input_size = args.size

# Style layer of interest
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_style_layers = len(style_layers)

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']
num_content_layers = len(content_layers)

seg_layers = style_layers


def main():
    print(args)

    for i in args.image:
        image_path = args.data_dir + str(i) + '.png'
        print(image_path)
        content_image = image_preprocessing(image_path, 'content', input_size, c=3)
        style_image = image_preprocessing(image_path, 'style', input_size, c=3)
        seg_image = image_preprocessing(image_path, 'segmentation', input_size, c=3)

        # From the segmentation image, extract a binary mask for each label.
        # Masks have 3 channels, are scaled as the input, value [0,1]. Saved in a list.
        seg_masks = extract_mask(seg_image, show=False)

        stylized_image, score = nst(content_image, style_image, seg_masks)

        file_name = args.save_dir + 'seg' + str(i) + '_' + str(int(np.round(score * 1e4))) + '.png'
        tensor_to_image(stylized_image).save(file_name)


def feature_extractor(inputs, layers):
    extractor = ContentModel(layers)
    features = extractor(inputs)
    return features


def nst(content_image, style_image, seg_masks):
    style_features = feature_extractor(style_image, style_layers)
    content_features = feature_extractor(content_image, content_layers)
    extractor = StyleSegContentModel(style_layers, seg_layers, content_layers)

    content_masks = []
    content_masks = resize_masks(seg_masks, content_features)
    style_masks = resize_masks(seg_masks, style_features)

    # ==================================================================================================================
    # Run gradient descent (with regularization term in the loss function)
    # ==================================================================================================================

    # Define a tf.Variable to contain the image to optimize
    stylized_image = tf.Variable(content_image)
    best_image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_seg_loss(outputs):
        style_outputs = outputs['style']
        style_loss = tf.add_n([tf.reduce_mean((gram_matrix(tf.multiply(style_outputs[name], mask[name])) -
                                               gram_matrix(tf.multiply(style_features[name], mask[name]))) ** 2)
                               for name in style_outputs.keys()
                               for mask in style_masks])
        style_loss *= style_weight / num_style_layers
        return style_loss

    def content_seg_loss(outputs):
        content_outputs = outputs['content']
        content_loss = tf.add_n([tf.reduce_mean(((tf.multiply(content_outputs[name], mask[name])) -
                                                 (tf.multiply(content_features[name], mask[name]))) ** 2)
                                 for name in content_outputs.keys()
                                 for mask in content_masks])
        content_loss *= content_weight / num_content_layers
        return content_loss

    def style_loss(outputs):
        style_outputs = outputs['style']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_features[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers
        return style_loss

    def content_loss(outputs):
        content_outputs = outputs['content']
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_features[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        return content_loss

    def style_content_loss(outputs):
        """Weighted combination of style and content loss"""
        loss = content_seg_loss(outputs) + style_loss(outputs)
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        image.assign(tf.repeat(tf.image.rgb_to_grayscale(image), 3, -1))

    start = time.time()
    score_max = 0

    for n in range(epochs):
        print("Epoch: {}".format(n), end='\t')
        for m in range(steps_per_epoch):
            train_step(stylized_image)
        mse_score = mse(pil_grayscale(stylized_image), pil_grayscale(style_image))
        psnr_score = psnr(pil_grayscale(stylized_image), pil_grayscale(style_image))
        ssim_score = tf.image.ssim(stylized_image, style_image, max_val=1.0).numpy()[0]
        print("\tMSE = {:.7f} \tPSNR = {:.7f}  \tSSIM = {:.7f}".format(mse_score, psnr_score, ssim_score))
        file_name = args.save_dir + 'opt/ep_' + str(n) + '.png'
        tensor_to_image(stylized_image).save(file_name)
        if ssim_score > score_max:
            score_max = ssim_score
            best_image = stylized_image

    end = time.time()
    print("Total time: {:.1f}\n".format(end - start))

    return best_image, score_max


def extract_mask(im, show=False):
    im = tensor_to_image(im).convert('L')           # 2D image
    w, h = im.size                                  # Get image dimensions
    pixels = list(im.getdata())                     # Get pixel list

    pixel_values = []                               # List of pixel values (only labels of interest)
    for pixel in pixels:
        if pixel not in pixel_values:
            pixel_values.append(pixel)
    pixel_values = [int(value) for value in pixel_values if 70 <= value <= 240]
    # print(sorted(pixel_values))

    masks = []                                                      # List of binary masks
    for value in pixel_values:
        mask = [1 if pixel == value else 0 for pixel in pixels]     # Create binary mask as a list
        mask = np.reshape(np.array(mask), [h, w])                   # Reshape as a 2D image (like im)
        if show:
            Image.fromarray(mask*255).show()
        mask = image_to_tensor(mask, c=3)                           # Redefine tensor
        masks.append(mask)

    return masks


def resize_masks(seg_masks, features):
    resized_masks = []
    for mask in seg_masks:
        mask = mask[:, :, :, 1]                         # Only 1 channel (there are 3 in total)
        mask = tf.expand_dims(mask, -1)                 # Redefine the tensor
        resized_mask = {}                               # Define the dict: {layer_name: mask}
        for name in features.keys():
            s = features[name].shape                    # Extract feature's shape
            if len(s) == 3:
                s = [1] + s                             # If only 3 dimensions, add the 4th at the beginning (=tensor)
            rm = tf.image.resize(mask, [s[1], s[2]])    # Same w, h as the features (resize method = bilinear)
            rm = tf.repeat(rm, s[3], -1)                # Same channels as the features
            resized_mask[name] = rm                     # Assign resized mask to dict label
        resized_masks.append(resized_mask)              # List of dicts
    return resized_masks


if __name__ == "__main__":
    main()
