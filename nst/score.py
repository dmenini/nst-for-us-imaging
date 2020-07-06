from PIL import Image
import tensorflow as tf
from utils import *

#tf.enable_eager_execution()

psnr_score, ssim_score = [], []

for i in range(1,2):
	mask = tf.expand_dims(tf_load_image('./output/masks/fg.png', 3), 0)
	stylized = tf.expand_dims(tf_load_image("/scratch/dmenini/nst-for-us-imaging/img/us_lq2hq_noisy_v2/"+ str(i)+".png", 3), 0)
	style = tf.expand_dims(tf_load_image("/scratch/dmenini/nst-for-us-imaging/img/hq_test/"+ str(i)+".png", 3), 0)

	h, w = stylized.shape[1], stylized.shape[2]

	#style = tf.image.resize(style, (h, w))
	#mask = tf.image.resize(mask, (h, w))

	style = tf.image.crop_to_bounding_box(style, 0, 193, 1000, 1000)
	mask = tf.image.crop_to_bounding_box(mask, 0, 193, 1000, 1000)

	stylized = tf.multiply(mask, stylized)
	style = tf.multiply(mask, style)

	mse_score = mse(stylized, style)
	psnr_score.append(tf.image.psnr(stylized, style, max_val=1.0).numpy()[0])
	ssim_score.append(tf.image.ssim(stylized, style, max_val=1.0).numpy()[0])

psnr = np.mean(np.array(psnr_score))
ssim = np.mean(np.array(ssim_score))
print("MSE = {:.7f} \tPSNR = {:.7f}  \tSSIM = {:.7f}".format(mse_score, psnr, ssim))