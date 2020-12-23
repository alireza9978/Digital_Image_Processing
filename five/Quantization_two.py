import cv2 as cv
import numpy as np


def quantize(array, quantize_level):
    my_array = (array.astype(np.float32) / 255) * (quantize_level - 1)
    my_array = np.round(my_array)
    my_array = (my_array * (255 / (quantize_level - 1))).astype(np.uint8)
    return my_array


pepper_image = cv.imread("../data/Pepper.bmp", cv.IMREAD_COLOR)
R = pepper_image[:, :, 2].astype(np.uint8)
G = pepper_image[:, :, 1].astype(np.uint8)
B = pepper_image[:, :, 0].astype(np.uint8)

quantized_R = quantize(R, 8)
quantized_G = quantize(G, 8)
quantized_B = quantize(B, 4)
new_images = np.dstack([quantized_B, quantized_G, quantized_R])
cv.imwrite("../output/pepper_quantized_8_8_4_level.jpg", new_images)
