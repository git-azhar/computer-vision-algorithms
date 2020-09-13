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


def get_harris_points(harris_image, min_dist: int = 10, threshold: float = 0.1):
    """
    :param harris_image: harris response image
    :param min_dist:  minimum number of pixels separating
                        corners and image boundary
    :param threshold:
    :return:  Return corners
    """

    # find top corner candidate above a threshold
    corner_thresh = harris_image.max() * threshold
    harris_image_t = (harris_image > corner_thresh) * 1

    # get coordinates of corner candidate & their values
    coordinates = np.array(harris_image_t.nonzero()).T
    candidate_values = [harris_image[c[0], c[1]] for c in coordinates]

    # sort
    index = np.argsort(candidate_values)

    # store allowed position in array
    allowed_positions = np.zeros(harris_image.shape)
    allowed_positions[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select best points taking min_dist into account
    filtered_coords = []
    for i in index:
        if allowed_positions[coordinates[i, 0], coordinates[i, 1]] == 1:
            filtered_coords.append(coordinates[i])
            allowed_positions[(coordinates[i, 0] - min_dist): (coordinates[i, 1] + min_dist),
             (coordinates[i, 1] - min_dist):(coordinates[i, 1] + min_dist)] = 0
    return filtered_coords
