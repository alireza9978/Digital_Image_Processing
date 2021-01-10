import cv2 as cv
import numpy as np


# کوچک کردن تصویر با حذف سطر و ستون
def down_sample_replication(input_image):
    down_sampled = np.zeros((int(input_image.shape[0] / 2), int(input_image.shape[1] / 2)),
                            dtype=np.uint8)
    for i in range(down_sampled.shape[0]):
        for j in range(down_sampled.shape[1]):
            down_sampled[i][j] = input_image[i * 2][j * 2]
    return down_sampled


def apply_filter(image, input_filter):
    filter_size = input_filter.shape[0]
    padding_size = (int(filter_size / 2), int(filter_size / 2))

    padded_image = np.pad(image, [padding_size, padding_size], mode='edge', )
    output_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(padded_image.shape[0] - input_filter.shape[0]):
        for j in range(padded_image.shape[1] - input_filter.shape[1]):
            image_sample = np.take(np.take(padded_image, range(i, i + input_filter.shape[0]), axis=0),
                                   range(j, j + input_filter.shape[1]), axis=1)
            temp_sum = 0
            for h in range(input_filter.shape[0]):
                for k in range(input_filter.shape[1]):
                    temp_sum += (input_filter[h][k] * image_sample[h][k])
            output_image[i][j] = abs(temp_sum)
    return output_image


def subtract(one, two):
    temp = (one.astype(np.int16) - two.astype(np.int16))
    temp = temp - temp.min()
    return ((temp / temp.max()) * 255).astype(np.uint8)


def decomposition(main_image, averaged_image):
    low_pass = subtract(main_image, averaged_image)
    new_image = subtract(main_image, low_pass)
    return new_image, low_pass


filter_size = 2
pyramid_level = 3

image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)

box_filter = np.ones([filter_size, filter_size], dtype=np.uint8) / (filter_size * filter_size)
temp_image = image
cv.imwrite("../output/pyramid_level_{}_main.jpg".format(0), temp_image)
i = 0
for i in range(pyramid_level):
    filtered_temp_image = apply_filter(temp_image, box_filter)
    temp_image, high_pass = decomposition(temp_image, filtered_temp_image)
    cv.imwrite("../output/pyramid_level_{}_main.jpg".format(i + 1), temp_image)
    cv.imwrite("../output/pyramid_level_{}_high_pass.jpg".format(i + 1), high_pass)
    temp_image = down_sample_replication(temp_image)

cv.imwrite("../output/pyramid_level_{}_main.jpg".format(pyramid_level + 1), temp_image)