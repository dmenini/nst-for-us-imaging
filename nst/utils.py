import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==================================================================================================================
#                                               MODEL UTILS
# ==================================================================================================================


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
        inputs = tf.keras.applications.vgg19.preprocess_input(inputs) 	# Subtract VGG_MEAN [103.939, 116.779, 123.68] to the channels, RGB->BGR. Not needed because inputs are BW.
        # inputs = inputs - 116.779
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def mse(result, true):
    result = tf.image.rgb_to_grayscale(result) * 255.0
    true = tf.image.rgb_to_grayscale(true) * 255.0
    return tf.reduce_mean((result - true)**2)


# ==================================================================================================================
#                                               IMAGE UTILS
# ==================================================================================================================

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor.astype(np.uint8))


def image_to_tensor(image, c):
    image = np.expand_dims(image, 0)
    tensor = np.expand_dims(image, -1)
    tensor = np.repeat(tensor, c, -1)
    return tf.constant(tensor)


def imgshow(image, c=3, title=None):
    if len(image.shape) > 2:
        image = tf.squeeze(image, axis=0)
        if c == 1:
            image = tf.squeeze(image, axis=-1)
    if title:
        plt.title(title)
    plt.imshow(image, cmap='gray')


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tf_load_image(path_to_image, c):
    img = tf.io.read_file(path_to_image)
    img = tf.image.decode_image(img, channels=c)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def scale_image(img, max_dim):
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape, method='nearest')
    img = img[tf.newaxis, :]

    img = tf.image.resize(img, [round(max_dim/1.386), max_dim], method='nearest')

    return img


def image_preprocessing(path_to_img, object, new_shape, c=3):
    """
    Read full image from file, decode as a tensor, value in range [0,1].
    Crop depending on the desired object.
    Scale image to the max dimension allowed. 
    """

    img = tf_load_image(path_to_img, c)

    (h, w, c) = img.shape

    if object == 'segmentation':
        img = tf.image.crop_to_bounding_box(img, 0, round(2 * w / 3), h, round(w / 3))
    elif object == 'content':
        img = tf.image.crop_to_bounding_box(img, 0, 0, h, round(w / 3))
    elif object == 'style':
        img = tf.image.crop_to_bounding_box(img, 0, round(w / 3), h, round(w / 3))
    elif object == 'clinical':
        img = tf.image.crop_to_bounding_box(img, 0, 20, h, w-20)
    else:
        print("Object must be either 'segmentation', 'content', 'style', 'clinical")
        return 1

    img = img[tf.newaxis, :]
    img = tf.image.resize(img, new_shape, method='nearest')

    return img
