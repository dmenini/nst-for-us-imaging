from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from PIL import Image

from nst_lib import *
from img_lib import *

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


def quick_nst(content_image, style_image):

    nst_module = tf.saved_model.load("./nst_model")

    stylized_image = nst_module(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image


def long_nst(content_image, style_image, reg=True):

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # print()
    # for layer in vgg.layers:
    #     print(layer.name)

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

    # Create a model (input = vgg's input, outputs = vgg's intermediate style outputs)
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    # ==================================================================================================================
    # Create a model that extracts both content and style
    # ==================================================================================================================

    # This model returns a dict of the gram matrix (style) of the style_layers and content of the content_layers
    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))

    style_results = results['style']

    # ==================================================================================================================
    # Run gradient descent (with regularization term in the loss function)
    # ==================================================================================================================

    # Set style and content target values
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # Define a tf.Variable to contain the image to optimize
    stylized_image = tf.Variable(content_image)

    # The paper recommends LBFGS, but Adam works okay too
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs, style_weight=1e-2, content_weight=1e4):
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
        total_variation_weight = 30
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            if reg:
                loss += total_variation_weight * total_variation_loss(image)  # tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()

    epochs = 10
    steps_per_epoch = 50

    step = 0
    for n in range(epochs):
        print("Epoch: {}".format(n))
        for m in range(steps_per_epoch):
            step += 1
            train_step(stylized_image)
            print(".", end='')
        #tensor_to_image(stylized_image)
        #imgshow(stylized_image, 'Stylized Image')
        file_name = 'img/opt/ep_' + str(n) + '.png'
        tensor_to_image(stylized_image).save(file_name)
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    return stylized_image


def main():

    content = 'segmentation'
    for i in range(8,10):
        image_path = 'img/data/new_att_all/' + str(i) + '.png'
        content_image, style_image = image_preprocessing(image_path, content=content)
        z = np.array_equal(content_image[0,:,:,0],content_image[0, :, :, 1])
        print(z)
        plt.subplot(1, 3, 1)
        imgshow(content_image, title='Content Image (' + str(content) + ')')
        plt.subplot(1, 3, 2)
        imgshow(style_image, title='Style Image (HQ)')

        #stylized_image = quick_nst(content_image, style_image)
        stylized_image = long_nst(content_image, style_image)

        plt.subplot(1, 3, 3)
        imgshow(rgb2gray(stylized_image), title='Stylized Image')

        plt.show(block=False)
        plt.pause(2)
        plt.close()

        file_name = 'img/result/' + str(i) + '.png'
        tensor_to_image(stylized_image).convert('LA').save(file_name)


if __name__ == "__main__":
    main()
