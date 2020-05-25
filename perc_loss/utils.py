import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np


# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename).convert('RGB')
    return img

# assumes data comes in batch form (ch, h, w)
def save_image(filename, data, norm):
    img = denormalize_tensor_transform(data, norm)
    img = Image.fromarray(img)
    img.save(filename)
    return img

# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

# using ImageNet values
def normalize_tensor_transform(norm):
    if norm: 
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    else:
        std = [1.0, 1.0, 1.0]
        mean = [0.0, 0.0, 0.0]
    return transforms.Normalize(mean=mean, std=std)


def denormalize_tensor_transform(data, norm):
    if norm:
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    else:
        std = np.array([1.0, 1.0, 1.0]).reshape((3, 1, 1))
        mean = np.array([0.0, 0.0, 0.0]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")

    return img


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


