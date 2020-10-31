import cv2 as cv
import numpy as np

levels = [8, 16, 32, 64, 128]


def make_histogram_pdf_cdf(input_image):
    histogram = np.zeros(256)
    total_count = input_image.shape[0] * input_image.shape[1]
    for row in input_image:
        for data in row:
            histogram[data] += 1
    pdf = histogram / total_count

    cdf = np.zeros(256)
    temp = 0
    for i in range(256):
        temp += pdf[i]
        cdf[i] = temp
    return histogram, pdf, cdf


def normal_hist(input_image, cdf, max_value):
    temp_cdf = np.round(cdf * max_value)
    normal_image = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            normal_image[i][j] = temp_cdf[input_image[i][j]]
    return normal_image


def quantization(input_image):
    outputs = []
    for level in levels:
        bit_count = int(np.log2(level))
        and_number = 0
        for i in range(8 - bit_count, 8):
            and_number += 2 ** i
        temp_image = np.zeros(input_image.shape, dtype=np.uint8)
        for i in range(input_image.shape[0]):
            for j in range(input_image.shape[1]):
                temp_image[i][j] = input_image[i][j] & and_number
        outputs.append(temp_image)
    return outputs


image = cv.imread('../data/Barbara.jpg', cv.IMREAD_GRAYSCALE)
image_histogram, pdf, cdf = make_histogram_pdf_cdf(image)
normal_image = normal_hist(image, cdf, 255)
normal_image_hist, normal_pdf, normal_cdf = make_histogram_pdf_cdf(normal_image)

quantized_images = quantization(image)
normal_quantized_images = quantization(normal_image)

mse = []
normal_mse = []
for i in range(len(levels)):
    mse.append(np.mean(np.square(np.subtract(image, quantized_images[i]))))
    normal_mse.append(np.mean(np.square(np.subtract(normal_image, normal_quantized_images[i]))))
    cv.imwrite("../output/quantized_level_{}.jpg".format(levels[i]), quantized_images[i])
    cv.imwrite("../output/normaled_quantized_level_{}.jpg".format(levels[i]), normal_quantized_images[i])

print("level", end=" ")
for i in range(len(levels)):
    print(levels[i], end=" ")
print()
print("without histeq", end=" ")
for i in range(len(levels)):
    print(mse[i], end=" ")
print()
print("with histeq", end=" ")
for i in range(len(levels)):
    print(normal_mse[i], end=" ")
