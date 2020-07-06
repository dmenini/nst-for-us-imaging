import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='File to read.')
parser.add_argument('--file', dest='file', default='eval_mix.sh.o4884230', type=str, help='Output file from gpu (perceptual loss)')

args = parser.parse_args()

file = open(args.file, 'r') 
lines = file.readlines() 

epoch=150
steps = 0
total_loss = []
style_loss = []
content_loss = []
total_loss_ = []
style_loss_ = []
content_loss_ = []

for line in lines:
	if "LOSS" in line:
		steps += 1
		l = line.split()
		total_loss.append(float(l[9]))
		style_loss.append(float(l[3]))
		content_loss.append(float(l[6]))

total_loss = np.array(total_loss)
style_loss = np.array(style_loss)
content_loss = np.array(content_loss)

for n in range(7):	
	total_loss_.append(total_loss[n*epoch:epoch*(n+1)])
	style_loss_.append(style_loss[n*epoch:epoch*(n+1)])
	content_loss_.append(content_loss[n*epoch:epoch*(n+1)])

for n in range(7):
	fig = plt.figure('loss' + str(n))
	plt.plot(total_loss_[n])
	plt.plot(content_loss_[n])
	plt.plot(style_loss_[n])
	plt.legend(['total', 'content', 'style'])
	plt.title('Loss')
	plt.savefig('loss' + str(n) + '.png')