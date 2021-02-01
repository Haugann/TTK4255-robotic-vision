import matplotlib.pyplot as plt
import numpy as np
from common import *

# Note: the sample image is naturally grayscale
I = rgb_to_gray(im2double(plt.imread('data/calibration.jpg')))

###########################################
#
# Task 3.1: Compute the Harris-Stephens measure
#
###########################################
sigma_D = 1
sigma_I = 3
alpha = 0.06

Ix, Iy, Im = derivative_of_gaussian(I, sigma_D)

# From Piazza:
# det(A) = A11 A22 - A12 A21
# trace(A) = A11 + A22

Ixy = gaussian(Ix*Iy, sigma_I)
Ixx = gaussian(Ix**2, sigma_I)
Iyy = gaussian(Iy**2, sigma_I)

det = Ixx*Iyy - 2*Ixy
trace = Ixx + Iyy

response = det - alpha*trace**2

###########################################
#
# Task 3.4: Extract local maxima
#
###########################################
corners_y, corners_x = extract_local_maxima(response, 0.001)

###########################################
#
# Figure 3.1: Display Harris-Stephens corner strength
#
###########################################
plt.figure(figsize=(13,5))
plt.imshow(response)
plt.colorbar(label='Corner strength')
plt.tight_layout()
# plt.savefig('plots/out_corner_strength.png', bbox_inches='tight', pad_inches=0) 

###########################################
#
# Figure 3.4: Display extracted corners
#
###########################################
plt.figure(figsize=(10,5))
plt.imshow(I, cmap='gray')
plt.scatter(corners_x, corners_y, linewidths=1, edgecolor='black', color='yellow', s=9)
plt.tight_layout()
# plt.savefig('plots/out_corners.png', bbox_inches='tight', pad_inches=0)

plt.show()
