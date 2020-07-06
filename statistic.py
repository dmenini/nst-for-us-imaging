from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open("lq2hq_mix0_34.png"))
content = np.array(Image.open("img/lq_dataset/new_att_all/645.png"))
style = np.array(Image.open("img/hq_dataset/new_att_all/645.png"))
style_masked = np.multiply(style, np.array(Image.open("output/masks/fg.png").convert('L')))

print(np.sum(img<10)/np.size(img), np.amax(img))
print(np.sum(content<10)/np.size(content))
print(np.sum(style<100)/np.size(style), np.amax(style))




