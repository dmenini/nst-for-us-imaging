from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sys
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
parser.add_argument('--weights', metavar='weights', type=float, nargs='+',
                    default=[1e-2, 1e4, 30], help='Style, content and total variation weights.')
parser.add_argument('--epochs', metavar='weights', type=int,
                    default=25, help='Max number of epochs.')
parser.add_argument('--steps', metavar='steps_per_epoch', type=int,
                    default=50, help='Number of steps per epoch.')
args = parser.parse_args()

style_weight = args.weights[0]
content_weight = args.weights[1]
total_variation_weight = args.weights[2]
epochs = args.epochs
steps_per_epoch = args.steps

# Style layer of interest
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_style_layers = len(style_layers)

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']
num_content_layers = len(content_layers)

seg_layers = style_layers


def main():
    print(args)

    for i in [1, 18, 34]:
        image_path = args.data_dir + str(i) + '.png'
        print(image_path)
        content_image = image_preprocessing(image_path, object='content', c=3)
        style_image = image_preprocessing(image_path, object='style', c=3)
        seg_image = image_preprocessing(image_path, object='segmentation', c=3)

        # From the segmentation image, extract a binary mask for each pixel value (a mask covers > 1% of the image)
        # Masks have 3 channels, are scaled as the input, value [0,1]. Saved in a list.
        seg_masks = extract_mask(seg_image, th=0.01)

        stylized_image, score = nst(content_image, style_image, seg_masks)

        file_name = args.save_dir + 'seg' + str(i) + '_' + str(int(np.round(score))) + '.png'
        pil_grayscale(stylized_image).save(file_name)


def feature_extractor(inputs, layers):
    extractor = ContentModel(layers)
    features = extractor(inputs)
    return features


def nst(content_image, style_image, seg_masks, reg=True):

    style_features = feature_extractor(style_image, style_layers)
    content_targets = feature_extractor(content_image, content_layers)
    extractor = StyleSegContentModel(style_layers, seg_layers, content_layers)

    resized_masks = []
    for mask in seg_masks:
        mask = mask[:, :, :, 1]             # only a channel
        mask = tf.expand_dims(mask, -1)     # redefine the tensor
        resized_mask = {}
        for name in style_features.keys():
            s = style_features[name].shape
            m = tf.image.resize(mask, [s[1], s[2]])     # same size as the style features
            m = tf.repeat(m, s[3], -1)                  # same channels as the style features
            resized_mask[name] = m
        resized_masks.append(resized_mask)      # list of dicts

    # resized_masks_content = []
    # for mask in seg_masks:
    #     mask = mask[:, :, :, 1]             # only a channel
    #     mask = tf.expand_dims(mask, -1)     # redefine the tensor
    #     resized_mask = {}
    #     for name in content_targets.keys():
    #         c = content_targets[name].shape
    #         m = tf.image.resize(mask, [c[1], c[2]])     # same size as the content targets
    #         m = tf.repeat(m, c[3], -1)                  # same channels as the content targets
    #         resized_mask[name] = m
    #     resized_masks_content.append(resized_mask)      # list of dicts

    # mask_features = [feature_extractor(mask, seg_layers) for mask in seg_masks]

    # ==================================================================================================================
    # Run gradient descent (with regularization term in the loss function)
    # ==================================================================================================================

    # Define a tf.Variable to contain the image to optimize
    stylized_image = tf.Variable(content_image)
    best_image = tf.Variable(content_image)
    h, w, c = content_image.shape[1], content_image.shape[2], content_image.shape[3]

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_seg_loss(outputs):
        style_outputs = outputs['style']
        style_loss = 0
        style_loss += tf.add_n([tf.reduce_mean((gram_matrix(tf.multiply(style_outputs[name], mask[name])) -
                                                gram_matrix(tf.multiply(style_features[name], mask[name]))) ** 2)
                                for name in style_outputs.keys()
                                for mask in resized_masks])
        style_loss *= style_weight / num_style_layers
        return style_loss

    def content_loss(outputs):
        content_outputs = outputs['content']
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        return content_loss

    def style_content_loss(outputs):
        """Weighted combination of style and content loss"""
        loss = style_seg_loss(outputs) + content_loss(outputs)
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            # image = concatenate_channels(image, seg_masks)
            loss = style_content_loss(outputs)
            if reg:
                loss += total_variation_weight * total_variation_loss(image)  # tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        return loss

    start = time.time()
    error_min = h * w * c   # Because the max difference is given when: (img_white - img_black)**2

    for n in range(epochs):
        print("Epoch: {}".format(n))
        for m in range(steps_per_epoch):
            loss = train_step(stylized_image)
            print(".", end='')
            # print('Loss = ', loss)
        mse_score = mse(pil_grayscale(stylized_image), pil_grayscale(style_image))
        psnr_score = psnr(pil_grayscale(stylized_image), pil_grayscale(style_image))
        print("\tMSE = {} \tPSNR = {}".format(mse_score, psnr_score))
        file_name = args.save_dir + 'opt/ep_' + str(n) + '.png'
        tensor_to_image(stylized_image).save(file_name)
        if mse_score < error_min:
            error_min = mse_score
            best_image = stylized_image

    end = time.time()
    print("Total time: {:.1f}\n".format(end - start))

    best_image = np.array(pil_grayscale(best_image))

    return best_image, mse_score


def concatenate_channels(image, masks):
    for mask in masks:
        image = tf.concat([image, mask], 3)
    return image


def extract_mask(image, th):
    image = np.array(image)
    image = np.squeeze(image)
    image = image[:, :, 0]
    pixel_val = []
    masks = []
    h, w = image.shape

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j] not in pixel_val) and image[i, j] != 0:
                pixel_val.append(image[i, j])
    # pixel_val.sort()

    for value in pixel_val:
        mask = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == value:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
        if np.sum(mask) > h * w * th:       # keep only masks with a decent coverage (coarse grain labels)
            mask = np.expand_dims(mask, 0)
            mask = np.expand_dims(mask, -1)
            mask = np.repeat(mask, 3, -1)
            # im = Image.fromarray(mask * 255)
            # im.show()
            masks.append(mask)
    return masks


if __name__ == "__main__":
    main()
