import cv2 as cv
import numpy as np


# کوچک کردن تصویر با حذف سطر و ستون
def down_sample_replication(input_image):
    down_sampled = np.zeros((int(input_image.shape[0] / 2), int(input_image.shape[1] / 2), input_image.shape[2]),
                            dtype=np.uint8)
    for i in range(down_sampled.shape[0]):
        for j in range(down_sampled.shape[1]):
            down_sampled[i][j] = input_image[i * 2][j * 2]
    return down_sampled


# بزرگ کردن تصویر با تکرار مقدار هر پیکسل
def up_sample_with_replication(input_image):
    output_image = np.zeros((input_image.shape[0] * 2, input_image.shape[1] * 2, input_image.shape[2]), dtype=np.uint8)
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i][j] = input_image[int(np.floor(i / 2))][int(np.floor(j / 2))]
    return output_image


def apply_filter(image, input_filter):
    filter_size = input_filter.shape[0]
    padding_size = (int(filter_size / 2), int(filter_size / 2))

    padded_image = np.pad(image, [padding_size, padding_size, (0, 0)], mode='edge', )
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


filter_size = 2
pyramid_level = 3

image = cv.imread("../data/Lena.bmp")

box_filter = np.ones([filter_size, filter_size], dtype=np.uint8) / (filter_size * filter_size)
new_image = image
for i in range(pyramid_level):
    new_image = apply_filter(new_image, box_filter)
    new_image = down_sample_replication(new_image)
    cv.imwrite("../output/pyramid_level_{}.jpg".format(i + 1), new_image)
for i in range(pyramid_level):
    new_image = up_sample_with_replication(new_image)
    cv.imwrite("../output/reconstructed_pyramid_level_{}.jpg".format(pyramid_level - (i + 1)), new_image)
