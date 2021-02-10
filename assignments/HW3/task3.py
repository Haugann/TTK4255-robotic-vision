import numpy as np
import matplotlib.pyplot as plt
from common import *

# Tip: Use np.loadtxt to load data into an array
K = np.loadtxt('data/task2K.txt')
X_o = np.loadtxt('data/task3points.txt')

T = translate(0,0,6)@rotate_x(15)@rotate_y(45)
X_c = T@X_o

u,v = project(K, X_c)

# You would change these to be the resolution of your image. Here we have
# no image, so we arbitrarily choose a resolution.
width,height = 600,400

#
# Figure for Task 2.2: Show pinhole projection of 3D points
#
plt.figure(figsize=(4,3))
plt.scatter(u, v, c='black', marker='.', s=20)

# The following commands are useful when the figure is meant to simulate
# a camera image. Note: these must be called after all draw commands!!!!

plt.axis('image')     # This option ensures that pixels are square in the figure (preserves aspect ratio)
                      # This must be called BEFORE setting xlim and ylim!
plt.xlim([0, width])
plt.ylim([height, 0]) # The reversed order flips the figure such that the y-axis points down
plt.savefig('plots/task3-2.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure
plt.show()
