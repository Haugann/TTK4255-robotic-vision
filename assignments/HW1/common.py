import numpy as np

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    return I.mean(axis=2) # Placeholder

def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    kernel = np.array([0.5, 0, -0.5])

    Ix = np.convolve(I.ravel(), kernel, mode='same').reshape(np.shape(I))
    Iy = np.convolve(I.ravel(order='F'), kernel, mode='same').reshape(np.shape(I), order='F')
    Im = np.zeros_like(I) # Placeholder
    # for i in range(I.shape[0]):
    #     row = np.convolve(I[i,:], kernel, mode='same')
    #     Ix[i] = row
    # for j in range(I.shape[1]):
    #     col = np.convolve(I[:,j], kernel, mode='same')
    #     Iy[:,j] = col
    Im = np.sqrt(np.power(Ix,2) + np.power(Iy,2))
    return Ix, Iy, Im

def g(x,sigma):
    return 1 / (2*np.pi*sigma**2) * np.exp(-x**2/(2*sigma**2))

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.
    if sigma == 0: 
        return I
    
    kernel = np.zeros(int(2*np.ceil(3*sigma)+1))
    for i in range(kernel.size):
        kernel[i] = g(i - kernel.size//2, sigma)

    result = np.convolve(I.ravel(), kernel, mode='same').reshape(np.shape(I))
    result = np.convolve(result.ravel(order='F'), kernel, mode='same').reshape(np.shape(I), order='F')

    return result / np.max(result)  # Normalize pixel values to remove odd
                                    # darkening issue.

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    theta = np.array([])
    edge_coords = np.nonzero(Im > threshold)
    for y, x in zip(edge_coords[0], edge_coords[1]):
        angle = np.arctan2(Iy[y,x], Ix[y,x])
        theta = np.append(theta, angle)

    return edge_coords[1], edge_coords[0], theta
