import cv2 as cv
import numpy as np

from three.util import apply_filter, median_filter

one = cv.imread("../output/Elaine_gaussian_01.jpg", cv.IMREAD_GRAYSCALE)
two = cv.imread("../output/Elaine_gaussian_05.jpg", cv.IMREAD_GRAYSCALE)
three = cv.imread("../output/Elaine_gaussian_10.jpg", cv.IMREAD_GRAYSCALE)

filters_size = [3, 5, 7, 9]
images = [one, two, three]
images_noise = [0.01, 0.05, 0.1]

for i in range(len(images)):
    image = images[i]
    for filter_size in filters_size:
        box_filter = np.ones([filter_size, filter_size], dtype=np.uint8) / (filter_size * filter_size)
        new_image = apply_filter(one, box_filter)
        cv.imwrite("../output/elaine_{}_fixed_gaussian_box_{}.jpg".format(images_noise[i], filter_size), new_image)
    for filter_size in filters_size:
        filtered_image = median_filter(image, filter_size)
        cv.imwrite("../output/elaine_{}_fixed_gaussian_median_{}.jpg".format(images_noise[i], filter_size),
                   filtered_image)
