import cv2 as cv
import numpy as np

from three.util import apply_filter

image = cv.imread("../data/Elaine.jpg", cv.IMREAD_GRAYSCALE)
filter_a = np.array([[1, 0, -1]], dtype=np.int8) * 0.5
filter_b = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.int8) * (1 / 6)
filter_c = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.int8) * (1 / 8)

filtered_a = apply_filter(image, filter_a)
filtered_b = apply_filter(image, filter_b)
filtered_c = apply_filter(image, filter_c)

cv.imwrite("../output/elaine_filtered_a.jpg", filtered_a)
cv.imwrite("../output/elaine_scaled_filtered_a.jpg", filtered_a * (255/np.max(filtered_a)))
cv.imwrite("../output/elaine_filtered_b.jpg", filtered_b)
cv.imwrite("../output/elaine_scaled_filtered_b.jpg", filtered_b * (255/np.max(filtered_b)))
cv.imwrite("../output/elaine_filtered_c.jpg", filtered_c)
cv.imwrite("../output/elaine_scaled_filtered_c.jpg", filtered_c * (255/np.max(filtered_c)))
