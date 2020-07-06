import numpy as np

x = np.random.randint(1,1000)
y = np.random.randint(1,10)

print("Quanto fa {}/{}?".format(x,y))
z = input()

if np.floor(float(x/y)) <= float(z) <= np.ceil(float(x/y)):
	print("Corretto!!")
else:
	print("Sbagliato tontolona. Fa {}".format(x/y))