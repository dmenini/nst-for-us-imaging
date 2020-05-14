from PIL import Image
import numpy as np

m = []

for i in range(35, 6600):
	img = np.array(Image.open('img/style_dataset/new_att_all/'+str(i)+'.png').convert('L'))
	m.append(np.mean(img))

result = sum(m)/len(m)
print(result)
