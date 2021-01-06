import cv2 as cv
import numpy as np
import pywt.data


def normalize(input_image):
    input_image = input_image - input_image.min()
    input_image = ((input_image / input_image.max()) * 255).astype(np.uint8)
    return input_image


pyramid_level = 3

# Load image
image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)

# Wavelet transform of image, and plot approximation and details
LL = image
for level in range(pyramid_level):
    LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    LL = normalize(LL)
    LH = normalize(LH)
    HL = normalize(HL)
    HH = normalize(HH)
    cv.imwrite("../output/wavelet_pyramid_level_{}_LL.jpg".format(level + 1), LL)
    cv.imwrite("../output/wavelet_pyramid_level_{}_LH.jpg".format(level + 1), LH)
    cv.imwrite("../output/wavelet_pyramid_level_{}_HL.jpg".format(level + 1), HL)
    cv.imwrite("../output/wavelet_pyramid_level_{}_HH.jpg".format(level + 1), HH)
