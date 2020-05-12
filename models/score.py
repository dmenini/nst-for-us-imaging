from PIL import Image
import tensorflow as tf
from utils import *

mask = tf.expand_dims(tf_load_image('./mask.png', 3), 0)
stylized = tf.expand_dims(tf_load_image("/scratch/dmenini/nst-for-us-imaging/img/gpu_result/12-05-20_15:52:39/seg34_423.png", 3), 0)
style = tf.expand_dims(tf_load_image('/scratch/dmenini/nst-for-us-imaging/img/style_dataset/new_att_all/34.png', 3), 0)

h, w = stylized.shape[1], stylized.shape[2]

mask = tf.image.resize(mask, (h, w))
style = tf.image.resize(style, (h, w))

stylized = tf.multiply(mask, stylized)
#style = tf.multiply(mask, style)

mse_score = mse(stylized, style)
psnr_score = tf.image.psnr(stylized, style, max_val=1.0).numpy()[0]
ssim_score = tf.image.ssim(stylized, style, max_val=1.0).numpy()[0]
print("MSE = {:.7f} \tPSNR = {:.7f}  \tSSIM = {:.7f}".format(mse_score, psnr_score, ssim_score))

#tensor_to_image(stylized).save('18:tv1_modified.png')