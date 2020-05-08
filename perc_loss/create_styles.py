from PIL import Image
from img_lib import *


for i in range(1,4000):
    image_path = 'img/data/new_att_all/' + str(i) + '.png'
    print(image_path)

    image = Image.open(image_path)
    w, h = image.size

    image = image.crop((round(w / 3), 0, round(2 * w / 3), h))

    file_name = 'img/style_dataset/new_att_all/' + str(i) + '.png'
    image.save(file_name)