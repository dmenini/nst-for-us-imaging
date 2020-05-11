from __future__ import division, print_function

import argparse
import numpy as np
import copy
import os
import time
import math
import tensorflow as tf

from vgg19 import Vgg19
from PIL import Image
from functools import partial


parser = argparse.ArgumentParser()
# Input Options
parser.add_argument('--data-dir',   dest='data_dir', nargs='?', type=str, default='img/data/new_att_all/',  help='Directory containing input images.')
parser.add_argument('--save-dir',   dest='save_dir', nargs='?', type=str, default='img/result/',            help='Directory where to store the results.')
parser.add_argument('--image',      dest='image',    nargs='?', type=int, default=1,                        help='Input image number.')
parser.add_argument("--serial",     dest='serial',   nargs='?', type=str, default='img/result/opt/',        help='Path to save the serial out_iter_X.png')

# Training Optimizer Options
parser.add_argument("--epochs",        dest='epochs',     nargs='?', type=int, default=1000, help='maximum image iteration')
parser.add_argument("--print_iter",    dest='print_iter', nargs='?', type=int, default=1,    help='print loss per iterations')
parser.add_argument("--save_iter",     dest='save_iter',  nargs='?', type=int, default=100,  help='save temporary result per iterations')
parser.add_argument("--lbfgs",         dest='lbfgs',      nargs='?',           default=False, help="True=lbfgs, False=Adam")

# Weight Options
parser.add_argument("--weights", dest='weights', nargs='+', type=float, default=[5e0, 1e2, 1e-3], help="weight of content loss, style loss, tv loss", )

args = parser.parse_args()

style_weight = args.weights[0]
content_weight = args.weights[1]
tv_weight = args.weights[2]

VGG_MEAN = [103.939, 116.779, 123.68]


def rgb2bgr(rgb, vgg_mean=True):
    if vgg_mean:
        return rgb[:, :, ::-1] - VGG_MEAN
    else:
        return rgb[:, :, ::-1]

def bgr2rgb(bgr, vgg_mean=False):
    if vgg_mean:
        return bgr[:, :, ::-1] + VGG_MEAN
    else:
        return bgr[:, :, ::-1]

def load_masks(seg, shape):
    def _extract_mask(seg, label):
        mask_0 = (np.array(seg)[:, :, 0] == label).astype(np.uint8)
        mask_1 = (np.array(seg)[:, :, 1] == label).astype(np.uint8)
        mask_2 = (np.array(seg)[:, :, 2] == label).astype(np.uint8)
        return np.multiply(np.multiply(mask_0, mask_1), mask_2).astype(np.float32)

    pixels = list(seg.getdata())              # Get pixel list
    labels = []                               # List of pixel values (only labels of interest)
    for pixel in pixels:
        if pixel not in labels:
            labels.append(pixel)
    labels = [label[0] for label in labels]
    labels = [int(value) for value in labels if 70 <= value <= 240]
    masks = [tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(seg, label)), 0), -1) for label in labels]

    # masks = [mask for mask in masks if np.sum(np.array(mask)) > 1386*1000*3*0.005]     # Filter out small masks to save memory for the GPU (max 10 masks allowed)
    masks = masks[0]
    return [masks]

def gram_matrix(activations):
    height = tf.shape(activations)[1]
    width = tf.shape(activations)[2]
    num_channels = tf.shape(activations)[3]
    gram_matrix = tf.transpose(activations, [0, 3, 1, 2])
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix

def content_loss(const_layer, var_layer, weight):
    return tf.reduce_mean(tf.math.squared_difference(const_layer, var_layer)) * weight

def style_loss(CNN_structure, const_layers, var_layers, content_segs, style_segs, weight):
    loss_styles = []
    layer_count = float(len(const_layers))
    layer_index = 0

    _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
    _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
    for layer_name in CNN_structure:
        layer_name = layer_name[layer_name.find("/") + 1:]

        # downsampling segmentation
        if "pool" in layer_name:
            content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(math.ceil(content_seg_height / 2))
            style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(math.ceil(style_seg_height / 2))

            for i in range(len(content_segs)):
                content_segs[i] = tf.compat.v1.image.resize_bilinear(content_segs[i], tf.constant((content_seg_height, content_seg_width)))
                style_segs[i] = tf.compat.v1.image.resize_bilinear(style_segs[i], tf.constant((style_seg_height, style_seg_width)))

        elif "conv" in layer_name:
            for i in range(len(content_segs)):
                # have some differences on border with torch
                content_segs[i] = tf.nn.avg_pool(tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                x = style_segs[i]
                style_segs[i] = tf.nn.avg_pool(tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                y = style_segs[i]
                print(tf.math.equal(x,y))

        if layer_name == var_layers[layer_index].name[var_layers[layer_index].name.find("/") + 1:]:
            print("Setting up style layer: <{}>".format(layer_name))
            const_layer = const_layers[layer_index]
            var_layer = var_layers[layer_index]

            layer_index = layer_index + 1

            layer_style_loss = 0.0
            for content_seg, style_seg in zip(content_segs, style_segs):
                gram_matrix_const = gram_matrix(tf.multiply(const_layer, style_seg))
                style_mask_mean   = tf.reduce_mean(style_seg)
                gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                        lambda: gram_matrix_const / (tf.cast(tf.size(const_layer), np.float32) * style_mask_mean),
                                        lambda: gram_matrix_const
                                    )

                gram_matrix_var   = gram_matrix(tf.multiply(var_layer, content_seg))
                content_mask_mean = tf.reduce_mean(content_seg)
                gram_matrix_var   = tf.cond(tf.greater(content_mask_mean, 0.),
                                        lambda: gram_matrix_var / (tf.cast(tf.size(var_layer), np.float32) * content_mask_mean),
                                        lambda: gram_matrix_var
                                    )

                diff_style_sum    = tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean

                layer_style_loss += diff_style_sum

            loss_styles.append(layer_style_loss * weight)
    return loss_styles

def total_variation_loss(output, weight):
    shape = output.get_shape()

    tv_loss = tf.reduce_sum((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
              (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight

def save_result(img_, str_):
    result = Image.fromarray(np.uint8(np.clip(img_, 0, 255.0)))
    result.save(str_)

def mse(result, true):
    result = tf.image.rgb_to_grayscale(result) * 255.0
    true = tf.image.rgb_to_grayscale(true) * 255.0
    return tf.reduce_mean(tf.squared_difference(result, true))

iter_count = 0
min_loss, best_image = float("inf"), None
def print_loss(args, loss_content, loss_styles_list, loss_tv, overall_loss, output_image):
    global iter_count, min_loss, best_image
    if iter_count % args.print_iter == 0:
        print('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, args.max_iter, loss_content))
        for j, style_loss in enumerate(loss_styles_list):
            print('\tStyle {} loss: {}'.format(j + 1, style_loss))
        print('\tTV loss: {}'.format(loss_tv))
        print('\tTotal loss: {}'.format(overall_loss - loss_affine))

    if overall_loss < min_loss:
        min_loss, best_image = overall_loss, output_image

    if iter_count % args.save_iter == 0 and iter_count != 0:
        save_result(best_image[:, :, ::-1], os.path.join(args.serial, 'out_iter_{}.png'.format(iter_count)))

    iter_count += 1

def stylize(args):
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    start = time.time()
    # prepare input images
    image_path = args.data_dir + '/' + str(args.image) + '.png'
    full_image = Image.open(image_path)
    w, h = full_image.size

    content_image = rgb2bgr(np.array(full_image.crop((0, 0, w/3, h)).convert("RGB"), dtype=np.float32))
    save_result(content_image, 'cont.png')

    width, height = content_image.shape[1], content_image.shape[0]
    content_image = content_image.reshape((1, height, width, 3)).astype(np.float32)

    style_image = rgb2bgr(np.array(full_image.crop((w/3, 0, w/3*2, h)).convert("RGB"), dtype=np.float32))
    style_image = style_image.reshape((1, height, width, 3)).astype(np.float32)

    seg = full_image.crop((w/3*2, 0, w, h)).convert("RGB").resize([width, height], resample=Image.BILINEAR)
 
    content_masks = load_masks(seg, [width, height])
    style_masks = content_masks

    init_image = np.random.randn(1, height, width, 3).astype(np.float32) * 0.0001
    # init_image = np.expand_dims(rgb2bgr(np.array(Image.open(args.init_image_path).convert("RGB"), dtype=np.float32)).astype(np.float32), 0)

    mean_pixel = tf.constant(VGG_MEAN)
    input_image = tf.Variable(init_image)

    with tf.name_scope("constant"):
        vgg_const = Vgg19()
        vgg_const.build(tf.constant(content_image))

        content_fv = sess.run(vgg_const.conv4_2)
        content_layer_const = tf.constant(content_fv)

        vgg_const.build(tf.constant(style_image))
        style_layers_const = [vgg_const.conv1_1, vgg_const.conv2_1, vgg_const.conv3_1, vgg_const.conv4_1, vgg_const.conv5_1]
        style_fvs = sess.run(style_layers_const)
        style_layers_const = [tf.constant(fv) for fv in style_fvs]

    with tf.name_scope("variable"):
        vgg_var = Vgg19()
        vgg_var.build(input_image)

    # which layers we want to use?
    style_layers_var = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
    content_layer_var = vgg_var.conv4_2

    # The whole CNN structure to downsample mask
    layer_structure_all = [layer.name for layer in vgg_var.get_all_layers()]

    # Content Loss
    loss_content = content_loss(content_layer_const, content_layer_var, float(content_weight))

    # Style Loss
    loss_styles_list = style_loss(layer_structure_all, style_layers_const, style_layers_var, content_masks, style_masks, float(style_weight))
    loss_style = 0.0
    for loss in loss_styles_list:
        loss_style += loss

    input_image_plus = tf.squeeze(input_image + mean_pixel, [0])

    # Total Variational Loss
    loss_tv = total_variation_loss(input_image, float(tv_weight))

    if args.lbfgs:
        overall_loss = loss_content + loss_tv + loss_style

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(overall_loss, method='L-BFGS-B', options={'maxiter': args.epochs, 'disp': 0})
        sess.run(tf.compat.v1.global_variables_initializer())
        print_loss_partial = partial(print_loss, args)
        optimizer.minimize(sess, fetches=[loss_content, loss_styles_list, loss_tv, overall_loss, input_image_plus], loss_callback=print_loss_partial)

        global min_loss, best_image, iter_count
        best_result = copy.deepcopy(best_image)
        min_loss, best_image = float("inf"), None
        return best_result
    else:
        VGGNetLoss = loss_content + loss_tv + loss_style
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-08)
        VGG_grads = optimizer.compute_gradients(VGGNetLoss, [input_image])

        train_op = optimizer.apply_gradients(VGG_grads)

        sess.run(tf.compat.v1.global_variables_initializer())
        min_loss, best_image = float("inf"), None
        for i in range(1, args.epochs):
            _, loss_content_, loss_styles_list_, loss_tv_, overall_loss_, output_image_ = sess.run([
                train_op, loss_content, loss_styles_list, loss_tv, VGGNetLoss, input_image_plus
            ])
            if i % args.print_iter == 0:
                print('Iteration {} / {}\n\tContent loss: {}'.format(i, args.epochs, loss_content_))
                for j, style_loss_ in enumerate(loss_styles_list_):
                    print('\tStyle {} loss: {}'.format(j + 1, style_loss_))
                print('\tTV loss: {}'.format(loss_tv_))
                print('\tTotal loss: {}'.format(overall_loss_ - loss_tv_))

            if overall_loss_ < min_loss:
                min_loss, best_image = overall_loss_, output_image_

            if i % args.save_iter == 0 and i != 0:
                save_result(best_image[:, :, ::-1], os.path.join(args.serial, '{}.png'.format(i)))

        return best_image

def main():
    print(args)
    best_image_bgr = stylize(args)
    result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
    # result = result.convert('L')
    result.save(args.save_dir + '/segmentation' + str(args.image) + '.png')

if __name__ == "__main__":
    main()