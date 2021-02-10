import numpy as np
import matplotlib.pyplot as plt
from common import *

# TASK 4.2
# Screw distance
d = 0.1145

im = plt.imread("data/quanser.jpg")
plt.xlim(200, 480)
plt.ylim(700, 420)
plt.imshow(im)


K = np.loadtxt('data/heli_k.txt')
T_platform_to_camera = np.loadtxt('data/platform_to_camera.txt')

x1 = np.array([0, 0, 0, 1])
x2 = np.array([d, 0, 0, 1])
x3 = np.array([0, d, 0, 1])
x4 = np.array([d, d, 0, 1])

X = np.stack((x1, x2, x3, x4), axis = 1)

X = T_platform_to_camera@X

u, v = project(K, X)
draw_frame(K,T_platform_to_camera, scale=d)

plt.scatter(u, v, c='yellow', marker='.', s=200)
plt.savefig("plots/task4-2", bbox_inches='tight', pad_inches=0) 
plt.show()


# TASK 4.3
def T_base_to_platform(psi):
	return translate(d/2,d/2,0) @ rotate_z(psi)


# TASK 4.4
def T_hinge_to_base(theta):
	return translate(0,0,0.325) @ rotate_y(theta)


# TASK 4.5
def T_arm_to_hinge():
	return translate(0,0,-0.05)


# TASK 4.6
def T_rotors_to_arm(phi):
	return translate(0.65, 0, -0.03) @ rotate_x(phi)


# TASK 4.7
points = np.loadtxt('data/heli_points.txt')
X_arm = points.T[:,:3]
X_rotors = points.T[:,3:]

im = plt.imread("data/quanser.jpg")
plt.imshow(im)

draw_frame(K, T_platform_to_camera, scale=0.05)
draw_frame(K, T_platform_to_camera@T_base_to_platform(11.6), scale=0.05)
draw_frame(K, T_platform_to_camera@T_base_to_platform(11.6)@T_hinge_to_base(28.9), scale=0.05)
draw_frame(K, T_platform_to_camera@T_base_to_platform(11.6)@T_hinge_to_base(28.9)@T_arm_to_hinge(), scale=0.05)
draw_frame(K, T_platform_to_camera@T_base_to_platform(11.6)@T_hinge_to_base(28.9)@T_arm_to_hinge()@T_rotors_to_arm(0), scale=0.05)

T_arm_to_camera = T_platform_to_camera@T_base_to_platform(11.6)@T_hinge_to_base(28.9)@T_arm_to_hinge()
T_rotors_to_camera = T_arm_to_camera@T_rotors_to_arm(0)

u,v = project(K, T_arm_to_camera@X_arm)
plt.scatter(u, v, c='yellow', marker='.', s=100)

u,v = project(K, T_rotors_to_camera@X_rotors)
plt.scatter(u, v, c='yellow', marker='.', s=100)

plt.savefig("plots/task4-7", bbox_inches='tight', pad_inches=0) 
plt.show()