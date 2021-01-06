import cv2 as cv
import numpy as np
import pywt.data


def normalize_and_quantize(input_image):
    return np.floor(input_image / 2) * 2


# Load image
image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)

# Wavelet transform of image, and plot approximation and details
LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
LL = normalize_and_quantize(LL)
LH = normalize_and_quantize(LH)
HL = normalize_and_quantize(HL)
HH = normalize_and_quantize(HH)
reconstruct_image = pywt.idwt2((LL, (LH, HL, HH)), 'haar').astype(np.uint8)
cv.imwrite("../output/quantized_wavelet.jpg", reconstruct_image)
cv.imwrite("../output/quantized_wavelet_LL.jpg", LL)
cv.imwrite("../output/quantized_wavelet_LH.jpg", LH)
cv.imwrite("../output/quantized_wavelet_HL.jpg", HL)
cv.imwrite("../output/quantized_wavelet_HH.jpg", HH)
PSNR = cv.PSNR(image, reconstruct_image)
print(PSNR)
