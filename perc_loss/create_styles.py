from PIL import Image


for i in range(1,6670):
    image_path = 'img/data/new_att_all/' + str(i) + '.png'
    print(image_path)

    image = Image.open(image_path)
    w, h = image.size

    image = image.crop((round(0), 0, round(w / 3), h))

    file_name = 'img/content_dataset/new_att_all/' + str(i) + '.png'
    image.save(file_name)