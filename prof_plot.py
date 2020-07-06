import matplotlib
import matplotlib.pyplot as plt
import numpy as np


x = np.array([200, 264.3, 318.9, 375.5, 428.9, 483.5, 538.6, 605.0, 669.3, 724.9, 785.3, 846.6, 1047.9, 1126.3])

fig = plt.figure('prof')
plt.plot(x)
plt.title('NST Timing')
plt.savefig('prof.png')