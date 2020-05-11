from PIL import Image
import tensorflow as tf
from utils import *

mask = Image.open('./mask.png').convert('L')
image = Image.open('/scratch/dmenini/nst-for-us-imaging/ep_250.png').convert('L')
style = Image.open('/scratch/dmenini/nst-for-us-imaging/img/style_dataset/new_att_all/1.png').convert('L')

mask = tf.constant(image_to_tensor(mask.resize([image.size[0], image.size[1]]),3) /255.0)
style = tf.constant(image_to_tensor(style.resize([image.size[0], image.size[1]]), 3)/255.0)
image = tf.constant(image_to_tensor(image, 3)/255.0)

result = tf.multiply(mask, image)
#style = tf.multiply(mask, style)

mse_score = mse(result, style)
psnr_score = tf.image.psnr(result, style, max_val=1.0).numpy()[0]
ssim_score = tf.image.ssim(result, style, max_val=1.0).numpy()[0]
print("MSE = {:.7f} \tPSNR = {:.7f}  \tSSIM = {:.7f}".format(mse_score, psnr_score, ssim_score))

tensor_to_image(result).save('1_modified.png')