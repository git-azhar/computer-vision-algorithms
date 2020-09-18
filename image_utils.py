import numpy as np


def append_images(img1: np.array, img2: np.array):
    """
    Return a new image that appends the two images side-by-side
    """

    # select image with fewest rows and fill in enough emtpy rows

    rows_1 = img1.shape[0]
    rows_2 = img2.shape[0]

    if rows_1 < rows_2:
        img1 = np.concatenate((img1, np.zeros((rows_2 - rows_1, img1.shape[1]))), axis=0)
    elif rows_2 < rows_1:
        img1 = np.concatenate((img2, np.zeros((rows_1 - rows_2, img2.shape[1]))), axis=0)

    # if rows are equal no need to fill.

    return np.concatenate((img1, img2), axis=1)


