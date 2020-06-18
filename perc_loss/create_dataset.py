from PIL import Image
from pathlib import Path
import numpy as np

for i in range(1,6670):
    image_path = 'img/data/new_att_all/' + str(i) + '.png'
    print(image_path)

    image = Image.open(image_path)
    w, h = image.size

    # content = np.expand_dims(np.array(image.crop((0, 0, round(w / 3), h))), axis=2)
    seg = np.expand_dims(np.array(image.crop((round(w/3*2), 0, w, h))), axis=2)

    content_seg = Image.fromarray(np.concatenate((seg, seg, seg), axis=2))

    Path("img/seg_dataset/new_att_all").mkdir(parents=True, exist_ok=True)

    file_name = 'img/seg_dataset/new_att_all/' + str(i) + '.png'
    content_seg.save(file_name)