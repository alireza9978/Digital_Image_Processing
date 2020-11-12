import cv2 as cv

from two.util import *
from two.util import make_histeq, make_pdf_cdf, make_histogram

images = [cv.imread("../data/HE1.jpg", cv.IMREAD_GRAYSCALE), cv.imread("../data/HE2.jpg", cv.IMREAD_GRAYSCALE),
          cv.imread("../data/HE3.jpg", cv.IMREAD_GRAYSCALE), cv.imread("../data/HE4.jpg", cv.IMREAD_GRAYSCALE)]
filer_size = [8, 32, 64]

for i in range(len(images)):
    image = images[i]
    image_histogram = make_histogram(image)
    image_pdf, image_cdf = make_pdf_cdf(image_histogram)
    image_histeq = make_histeq(image, image_cdf, 255)
    for size in filer_size:
        new_image = local_histogram_equalization_two(image, size)
        new_image_limit = local_histogram_equalization_three(image, size)

        # create a CLAHE object (Arguments are optional).
        cl1 = cv.createCLAHE(clipLimit=2.0, tileGridSize=(size, size)).apply(image)

        cv.imwrite('../output/HE{}_filter_{}_cv_method.jpg'.format(i, size), cl1)
        cv.imwrite("../output/HE{}_filter_{}_local_limit.jpg".format(i, size), new_image_limit)
        cv.imwrite("../output/HE{}_filter_{}_local.jpg".format(i, size), new_image)
    cv.imwrite("../output/HE{}_global.jpg".format(i), image_histeq)
