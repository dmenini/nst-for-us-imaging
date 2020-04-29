from PIL import Image

Image.open('../img/result/ep_6.png').convert('L').save('../img/result/seg_34_142.png')