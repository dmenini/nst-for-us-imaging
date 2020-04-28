import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

num = 2

image_path = 'img/data/new_att_all/'+str(num)+'.png'
img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

h,w = img.shape
crop_img = img[0 : h, round(w/3) : round(2*w/3)]

hist = cv2.calcHist([crop_img],[0],None,[256],[0,256])
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

image_path = 'img/result/'+str(num)+'.png'
img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

h,w = img.shape
hist = cv2.calcHist(img,[0],None,[256],[0,256])
plt.plot(hist)
plt.xlim([0, 256])
plt.show()