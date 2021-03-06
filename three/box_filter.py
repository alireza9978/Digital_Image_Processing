import cv2 as cv
import numpy as np

from three.util import apply_filter

image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)
filter_size = 3
box_filter = np.ones([filter_size, filter_size], dtype=np.uint8) / (filter_size * filter_size)
new_image = image
for i in range(100):
    new_image = apply_filter(new_image, box_filter)
    if (i + 1) % 10 == 0:
        cv.imwrite("../output/filtered_{}_time.jpg".format(i + 1), new_image)
