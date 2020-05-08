import os
import tensorflow as tf

import numpy as np
import time
import inspect

class Vgg19:
    def __init__(self):
        # if vgg19_npy_path is None:
        #     path = inspect.getfile(Vgg19)
        #     path = os.path.abspath(os.path.join(path, os.pardir))
        #     path = os.path.join(path, "vgg19.npy")
        #     vgg19_npy_path = path
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        self.data_dict = None
        #self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

    def build(self, bgr, clear_data=True):
        """
        load variable from npy to build the VGG
        """
        self.conv1_1 = self.conv_layer(bgr, ("block1_conv1"))
        self.conv1_2 = self.conv_layer(self.conv1_1, ("block1_conv2"))
        self.pool1 = self.max_pool(self.conv1_2, ('block1_pool'))

        self.conv2_1 = self.conv_layer(self.pool1, ("block2_conv1"))
        self.conv2_2 = self.conv_layer(self.conv2_1, ("block2_conv2"))
        self.pool2 = self.max_pool(self.conv2_2, ('block2_pool'))

        self.conv3_1 = self.conv_layer(self.pool2, ("block3_conv1"))
        self.conv3_2 = self.conv_layer(self.conv3_1, ("block3_conv2"))
        self.conv3_3 = self.conv_layer(self.conv3_2, ("block3_conv3"))
        self.conv3_4 = self.conv_layer(self.conv3_3, ("block3_conv4"))
        self.pool3 = self.max_pool(self.conv3_4, ('block3_pool'))

        self.conv4_1 = self.conv_layer(self.pool3, ("block4_conv1"))
        self.conv4_2 = self.conv_layer(self.conv4_1, ("block4_conv2"))
        self.conv4_3 = self.conv_layer(self.conv4_2, ("block4_conv3"))
        self.conv4_4 = self.conv_layer(self.conv4_3, ("block4_conv4"))
        self.pool4 = self.max_pool(self.conv4_4, ('block4_pool'))

        self.conv5_1 = self.conv_layer(self.pool4, ("block5_conv1"))
        #self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        #self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        #self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        #self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        if clear_data:
            self.data_dict = None

    def get_all_layers(self):
        return [self.conv1_1, self.conv1_2, self.pool1,\
                self.conv2_1, self.conv2_2, self.pool2, \
                self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4, self.pool3, \
                self.conv4_1, self.conv4_2, self.conv4_3, self.conv4_4, self.pool4, \
                self.conv5_1]

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.vgg.get_layer(name).get_weights()[0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.vgg.get_layer(name).get_weights()[1], name="biases")