from scipy.ndimage import filters
import numpy as np


def compute_harris_response(image, sigma: int = 5) -> float:
    """ Compute Harris corner detection response function for each pixel in an image (grayscale)"""

    # derivatives
    im_x = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (0, 1), im_x)
    im_y = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (1, 0), im_y)

    # Compute Components of Harris Matrix
    W_xx = filters.gaussian_filter(im_x * im_x, sigma)
    W_xy = filters.gaussian_filter(im_x * im_y, sigma)
    W_yy = filters.gaussian_filter(im_y * im_y, sigma)

    # determinant and trace

    W_det = W_xx * W_yy - W_xy ** 2
    W_trace = W_xx + W_yy

    return W_det / W_trace
