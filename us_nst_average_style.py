from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib as mpl
import time
import pickle

from nst_lib import *
from img_lib import *

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30
epochs = 25
steps_per_epoch = 50
CREATE = 1

# Style layer of interest
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_style_layers = len(style_layers)

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']
num_content_layers = len(content_layers)


def style_processing(style_image):
    # This model returns a dict of the gram matrix (style) of the style_layers
    style_extractor = StyleModel(style_layers)

    style_target = style_extractor(style_image)

    return style_target


def content_processing(content_image):
    # This model returns a dict of content of the content_layers
    content_extractor = ContentModel(content_layers)

    content_target = content_extractor(content_image)

    return content_target


def nst(content_image, style_image, content_target, style_target, reg=True):

    extractor = StyleContentModel(style_layers, content_layers)

    # ==================================================================================================================
    # Run gradient descent (with regularization term in the loss function)
    # ==================================================================================================================

    # Define a tf.Variable to contain the image to optimize
    stylized_image = tf.Variable(content_image)
    best_image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs):
        """Weighted combination of style and content loss"""
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            if reg:
                loss += total_variation_weight * total_variation_loss(image)  # tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()
    error_min = 10000

    step = 0
    for n in range(epochs):
        print("Epoch: {}".format(n))
        for m in range(steps_per_epoch):
            step += 1
            train_step(stylized_image)
            print(".", end='')
        mse_score = mse(pil_grayscale(stylized_image), pil_grayscale(style_image))
        psnr_score = mse(pil_grayscale(stylized_image), pil_grayscale(style_image))
        ssim_score = ssim(pil_grayscale(stylized_image), pil_grayscale(style_image))
        print("\tMSE = {} \tPSNR = {} \tSSIM = {}".format(mse_score, psnr_score, ssim_score))
        file_name = 'img/opt/ep_' + str(n) + '.png'
        tensor_to_image(stylized_image).save(file_name)
        if mse_score < error_min:
            error_min = mse_score
            best_image = stylized_image

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    return best_image, mse_score


def average_style(style_targets):
    n = len(style_targets)
    avg_style = {key: None for key in style_targets[0]}
    for key in style_targets[0]:
        temp = np.zeros_like(style_targets[0][key])
        for target in style_targets:
            temp = temp + target[key]
        avg_style[key] = temp / n
    return avg_style


def main():

    if CREATE:
        style_targets = []
        for i in range(1, 68):
            image_path = 'img/data/new_att_all/' + str(i) + '.png'
            style_image = image_preprocessing(image_path, object='style', c=3)
            style_targets.append(style_processing(style_image))

        print('Averaged the style over {} images'.format(len(style_targets)))
        # style_targets is a list of dicts of 3D arrays

        style_target = average_style(style_targets)

        with open('avg_style.pickle', 'wb') as handle:
            pickle.dump(style_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        filename = 'avg_style.pickle'
        print('Loading dict from {}'.format(filename))
        with open(filename, 'rb') as handle:
            style_target = pickle.load(handle)

    for j in [1, 18, 34]:

        image_path = 'img/data/new_att_all/' + str(j) + '.png'
        content_image = image_preprocessing(image_path, object='content', c=3)
        style_image = image_preprocessing(image_path, object='style', c=3)
        content_target = content_processing(content_image)

        stylized_image, score = nst(content_image, style_image, content_target, style_target, reg=True)

        file_name = 'img/result/avg_' + str(j) + '_' + str(score) + '.png'
        pil_grayscale(stylized_image).save(file_name)


if __name__ == "__main__":
    main()
