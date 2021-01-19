# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np

# %% TASK 2.1
grass = plt.imread("data/grass.jpg")
h,w = grass.shape[:2]

print("Image height: {}\nImage width: {}".format(h,w))

# %% TASK 2.2
fig1, axs = plt.subplots(1, 3, figsize=(13,3))
fig1.suptitle('RGB-channelled images')
for ax, i in zip(axs, range(3)):
	ax.imshow(grass[:,:,i])
	ax.set_title("Channel {}".format(i+1))

plt.savefig("figures/RGB-channels.jpg")

# %% TASK 2.3
grass_green_thresholded = grass[:,:,1] > 190
plt.imshow(grass_green_thresholded)
plt.savefig("figures/thresholded_grass.jpg")

# %% TASK 2.4
r = grass[:,:,0] / grass.sum(axis=2)
g = grass[:,:,1] / grass.sum(axis=2)
b = grass[:,:,2] / grass.sum(axis=2)

fig2, axs = plt.subplots(1, 3, figsize=(13,3))
fig2.suptitle('RGB-channelled images')
for ax, img, name in zip(axs, [r,g,b], ["R","G","B"]):
	ax.imshow(img)
	ax.set_title("Channel: {}".format(name))

plt.savefig("figures/normalized-RGB-channels.jpg")

# %% TASK 2.5
grass_green_thresholded = g > 0.4
plt.imshow(grass_green_thresholded, cmap='gray')
plt.savefig("figures/normalized_thresholded_grass.jpg")
# %%
