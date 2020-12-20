import cv2 as cv
import numpy as np


def quantize(array, quantize_level):
    my_array = (array.astype(np.float32) / 255) * quantize_level
    my_array = np.round(my_array)
    my_array = (my_array * (255 / (quantize_level - 1))).astype(np.uint8)
    return my_array


levels = [64, 32, 16, 8]

pepper_image = cv.imread("../data/Pepper.bmp", cv.IMREAD_COLOR)
R = pepper_image[:, :, 2].astype(np.uint8)
G = pepper_image[:, :, 1].astype(np.uint8)
B = pepper_image[:, :, 0].astype(np.uint8)

errors = []
for level in levels:
    quantized_R = quantize(R, level)
    quantized_G = quantize(G, level)
    quantized_B = quantize(B, level)
    new_images = np.dstack([quantized_B, quantized_G, quantized_R])
    cv.imwrite("../output/pepper_quantized_{}_level.jpg".format(level), new_images)
    MSE = np.mean(np.square(np.subtract(pepper_image, new_images)))
    PSNR = cv.PSNR(pepper_image, new_images)
    errors.append((MSE, PSNR, level))

print(errors)
