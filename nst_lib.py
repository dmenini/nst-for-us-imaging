import tensorflow as tf
import numpy as np
from img_lib import *


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# class StyleSegContentModel(tf.keras.models.Model):
#     def __init__(self, style_layers, content_layers, resized_masks):
#         super(StyleSegContentModel, self).__init__()
#         self.vgg = vgg_layers(style_layers + content_layers)
#         self.style_layers = style_layers
#         self.content_layers = content_layers
#         self.num_style_layers = len(style_layers)
#         self.vgg.trainable = False
#
#     def call(self, inputs):
#         """Expects float input in [0,1]"""
#         inputs = inputs * 255.0
#         preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
#         outputs = self.vgg(preprocessed_input)
#         style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
#
#
#
#
#         content_dict = {content_name: value
#                         for content_name, value
#                         in zip(self.content_layers, content_outputs)}
#         style_dict = {style_name: value
#                       for style_name, value
#                       in zip(self.style_layers, style_outputs)}
#
#         for mask in self.resized_masks:
#             image = tf.concat([image, mask], 3)
#
#         return {'content': content_dict, 'style': style_dict, 'seg': seg_dict}


class StyleModel(tf.keras.models.Model):
    def __init__(self, style_layers):
        super(StyleModel, self).__init__()
        self.vgg = vgg_layers(style_layers)
        self.style_layers = style_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs = outputs[:self.num_style_layers]

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        style_dict = {style_name: value
                      for style_name, value in zip(self.style_layers, style_outputs)}

        return style_dict


class ContentModel(tf.keras.models.Model):
    def __init__(self, content_layers):
        super(ContentModel, self).__init__()
        self.vgg = vgg_layers(content_layers)
        self.content_layers = content_layers
        self.num_content_layers = len(content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        content_outputs = outputs[:self.num_content_layers]
        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_outputs)}

        return content_dict


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    """ Calculate style by means of Gram Matrix, which include means and correlation across feature maps"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var


def total_variation_loss(image):
    """Regularizaton term on the high frequency components"""
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def mse(result, true):
    result = np.array(result)
    true = np.array(true)

    n = result.shape[0] * result.shape[1]
    mse = 0

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            mse = int(mse) + (int(result[i, j]) - int(true[i, j])) ** 2

    return mse / n
