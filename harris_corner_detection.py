import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import filters


def compute_harris_response(image, sigma: int = 5):
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
    :param threshold:
    :param harris_image: harris response image
    :param min_dist:  minimum number of pixels separating
                        corners and image boundary
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
    filtered_coordinates = []
    for i in index:
        if allowed_positions[coordinates[i, 0], coordinates[i, 1]] == 1:
            filtered_coordinates.append(coordinates[i])
            allowed_positions[(coordinates[i, 0] - min_dist): (coordinates[i, 1] + min_dist),
            (coordinates[i, 1] - min_dist):(coordinates[i, 1] + min_dist)] = 0
    return filtered_coordinates


def plot_harris_points(image: np.array, filtered_coordinates: list):
    """
    Plots corners found in image
    """
    print('plotting')
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coordinates], [p[0] for p in filtered_coordinates], 'o')
    plt.axis('off')
    plt.show()


def get_descriptors(image: np.array, filtered_coordinates: list, width: int = 5):
    """
    For each point return pixel value around the point using a neighbourhood of width (2*width+1)
    """
    descs = []
    for coords in filtered_coordinates:
        patch = image[coords[0] - width: coords[0] + width + 1, coords[1] - width: coords[1] + width + 1].flatten()
        descs.append(patch)
    return descs


def match(desc1, desc2, thresh: float = 0.5):
    """
    for each descriptor in the first image select its match to second image using normalized cross correlation
    """
    n = len(desc1[0])

    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1)) / np.std(desc1)
            d2 = (desc2[i] - np.mean(desc2)) / np.std(desc2)
            ncc_value = np.sum(d1 * d2) / (n - 1)
            if ncc_value > thresh:
                d[i, j] = ncc_value
    ndx = np.argsort(-d)
    match_scores = ndx[:, 0]
    return match_scores


def match_two_sided(desc1, desc2, thresh: float = 0.5):
    """
    match from the second image to the first and filter out the matches that are not the best both ways. This
    function does just that.
    """
    matches_12 = match(desc1, desc2, thresh)
    matches_21 = match(desc2, desc1, thresh)

    ndx_12 = np.where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12 = -1
    return matches_12


img = np.array(Image.open('images/house.jpg').convert('L'))
harris_img = compute_harris_response(img)
filtered_coords = get_harris_points(harris_img, 6)
plot_harris_points(img, filtered_coords)


