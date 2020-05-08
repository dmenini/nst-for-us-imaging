import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename)
    img = np.expand_dims(img,2)
    img = np.repeat(img, 3, 2)
    return img

# assumes data comes in batch form (ch, h, w)
def save_image(filename, data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

# using ImageNet values
def normalize_tensor_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def crop_image(filepath, object):
    image = load_image(filepath)
    h, w = image.size

    if object == 'content':
        image.crop((0, h, round(w / 3), 0))
    elif object == 'style':
        image.crop((round(w / 3), h, round(2 * w / 3), 0))
    elif object == 'segmentation':
        image = tf.image.crop_to_bounding_box((round(2 * w / 3), h, w, 0))
    else:
        print("Object must be either 'segmentation', 'content' or 'style'")
        return 1

    return image