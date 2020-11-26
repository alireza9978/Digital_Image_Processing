import cv2 as cv
import numpy as np

from three.util import apply_filter, median_filter

main_image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)

one = cv.imread("../output/lena_gaussian_01.jpg", cv.IMREAD_GRAYSCALE)
two = cv.imread("../output/lena_gaussian_05.jpg", cv.IMREAD_GRAYSCALE)
three = cv.imread("../output/lena_gaussian_10.jpg", cv.IMREAD_GRAYSCALE)

filters_size = [3, 5, 7, 9]
images = [one, two, three]
images_noise = [0.01, 0.05, 0.1]

result_box = []
result_median = []

for i in range(len(images)):
    image = images[i]
    this_noise_result_box = []
    this_noise_result_median = []
    for filter_size in filters_size:
        box_filter = np.ones([filter_size, filter_size], dtype=np.uint8) / (filter_size * filter_size)
        new_image = apply_filter(image, box_filter)
        this_noise_result_box.append(np.mean(np.square(np.subtract(main_image, new_image))))
        cv.imwrite("../output/lena_{}_fixed_gaussian_box_{}.jpg".format(images_noise[i], filter_size), new_image)
    for filter_size in filters_size:
        filtered_image = median_filter(image, filter_size)
        this_noise_result_median.append(np.mean(np.square(np.subtract(main_image, filtered_image))))
        cv.imwrite("../output/lena_{}_fixed_gaussian_median_{}.jpg".format(images_noise[i], filter_size),
                   filtered_image)
    result_box.append(this_noise_result_box)
    result_median.append(this_noise_result_median)

print(result_box)
print(result_median)
