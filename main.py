import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from harris import compute_harris_response, get_harris_points,\
    get_descriptors, match_two_sided, plot_matches

img1 = np.array(Image.open('images/house.jpg').convert('L'))
img2 = np.array(Image.open('images/house.jpg').convert('L'))
wid = 100

harris_img = compute_harris_response(img1, 5)
filtered_coords1 = get_harris_points(harris_img, wid + 1)
d1 = get_descriptors(img1, filtered_coords1, wid)

harris_img = compute_harris_response(img2, 5)
filtered_coords2 = get_harris_points(harris_img, wid + 1)
d2 = get_descriptors(img2, filtered_coords2, wid)

print('[INFO] Starting Matching ')
matches = match_two_sided(d1, d2)
print('[INFO] match complete')

plt.figure()
plt.gray()
plot_matches(img1, img2, filtered_coords1, filtered_coords2, matches)
plt.show()
