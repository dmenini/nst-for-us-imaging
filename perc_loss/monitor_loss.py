import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='File to read.')
parser.add_argument('--file', dest='file', default='eval_perc.sh.o4852014', type=str, help='Output file from gpu (perceptual loss)')

args = parser.parse_args()


file = open(args.file, 'r') 
lines = file.readlines() 

steps = 0
total_loss = []
style_loss = []
content_loss = []  

for line in lines:
	if "Epoch" in line:
	    l = line.split()
	    epoch = int(l[1][:-1])
	    total_loss.append(float(l[5]))
	    style_loss.append(float(l[7]))
	    content_loss.append(float(l[9]))

total_loss = np.array(total_loss)
style_loss = np.array(style_loss)
content_loss = np.array(content_loss)
x = np.arange(0,np.size(total_loss)) / (epoch+1)

fig = plt.figure('Perceptual Loss')
plt.plot(x, total_loss)
plt.plot(x, content_loss)
plt.plot(x, style_loss)
plt.legend(['total', 'content', 'style'])
plt.title('Perceptual Loss')
plt.savefig('perc_loss.png')
