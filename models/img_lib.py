import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 1386
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    print('Input shape:', img.shape)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    print('Input shape post resizing:', img.shape)
    return img


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


def read_image_as_tensor(path_to_img, c=1):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=c)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def scale_image(img):
    max_dim = 1386
    #print('Input shape pre resizing:', img.shape)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    #print('Input shape post resizing:', img.shape)
    return img


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def image_preprocessing(image_path, object, c=3):
    image = read_image_as_tensor(image_path, c=c)
    (h, w, c) = image.shape

    if object == 'segmentation':
        image = tf.image.crop_to_bounding_box(image, 0, round(2 * w / 3), h, round(w / 3))
    elif object == 'content':
        image = tf.image.crop_to_bounding_box(image, 0, 0, h, round(w / 3))
    elif object == 'style':
        image = tf.image.crop_to_bounding_box(image, 0, round(w / 3), h, round(w / 3))
    else:
        print("Object must be either 'segmentation', 'content' or 'style'")
        return 1

    image = scale_image(image)

    return image


def pil_grayscale(image):
    return tensor_to_image(image).convert('L')