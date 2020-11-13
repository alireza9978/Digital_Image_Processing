import cv2 as cv
import numpy as np

from three.util import apply_filter

image = cv.imread("../data/Elaine.jpg", cv.IMREAD_GRAYSCALE)
filter_size = 3
box_filter = np.ones([filter_size, filter_size], dtype=np.uint8)
new_image = image
for i in range(10):
    new_image = apply_filter(new_image, box_filter)
    if (i + 1) % 5 == 0:
        cv.imwrite("../output/filtered_{}_time.jpg".format(i + 1), new_image)
