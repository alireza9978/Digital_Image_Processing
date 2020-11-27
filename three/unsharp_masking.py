import cv2 as cv

from three.util import *


def make_filter(filter_size):
    x, y = np.meshgrid(np.linspace(-1, 1, filter_size), np.linspace(-1, 1, filter_size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    my_filter = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return my_filter / my_filter.sum()


image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE).astype(np.uint8)
filter_sizes = [3, 5, 7, 9]
alpha = np.linspace(0, 1, 5)
for size in filter_sizes:
    temp_filter = make_filter(size)
    filtered_image = apply_filter(image, temp_filter)
    for a in alpha:
        result = np.subtract(np.multiply(1 - a, image), np.multiply(a, filtered_image))
        cv.imwrite("../output/lena_gaussian_{}_alpha_{}.jpg".format(size, a), result)
