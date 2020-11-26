import cv2 as cv
import numpy as np

from three.util import apply_filter

image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)
filter_a = np.array([[1, 0], [0, -1]], dtype=np.int8)
filter_b = np.array([[0, 1], [-1, 0]], dtype=np.int8)

filtered_a = apply_filter(image, filter_a)
filtered_b = apply_filter(image, filter_b)

cv.imwrite("../output/lena_robert_filtered_a.jpg", filtered_a)
cv.imwrite("../output/lena_robert_scaled_filtered_a.jpg", filtered_a * (255 / np.max(filtered_a)))
cv.imwrite("../output/lena_robert_filtered_b.jpg", filtered_b)
cv.imwrite("../output/lena_robert_scaled_filtered_b.jpg", filtered_b * (255 / np.max(filtered_b)))
