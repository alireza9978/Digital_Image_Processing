import cv2 as cv
import numpy as np
import pywt


def quantize(input_image):
    return np.floor(input_image / 2) * 2


# Load image
image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)
cv.imwrite("../output/lena_gray.jpg", image)

pyramid_level = 3

# Wavelet transform of image, and plot approximation and details
OLD_LL = image
for i in range(pyramid_level):
    LL, (LH, HL, HH) = pywt.dwt2(OLD_LL, 'haar')
    TEMP_LL = LL
    LL = quantize(LL)
    LH = quantize(LH)
    HL = quantize(HL)
    HH = quantize(HH)
    reconstruct_image = pywt.idwt2((LL, (LH, HL, HH)), 'haar').astype(np.uint8)
    cv.imwrite("../output/quantized_and_reconstructed_wavelet_level_{}.jpg".format(i + 1), reconstruct_image)
    PSNR = cv.PSNR(OLD_LL, reconstruct_image)
    print("level_{}_image_size_{}".format(i + 1, OLD_LL.shape[0]), PSNR)
    TEMP_LL = TEMP_LL - TEMP_LL.min()
    OLD_LL = ((TEMP_LL / TEMP_LL.max()) * 255).astype(np.uint8)
