import cv2 as cv
import numpy as np


def make_filter(filter_size):
    x, y = np.meshgrid(np.linspace(-1, 1, filter_size), np.linspace(-1, 1, filter_size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    my_filter = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return my_filter / my_filter.sum()


# کوچک کردن تصویر با حذف سطر و ستون
def down_sample_replication(input_image):
    down_sampled = np.zeros((int(input_image.shape[0] / 2), int(input_image.shape[1] / 2), input_image.shape[2]),
                            dtype=np.uint8)
    for i in range(down_sampled.shape[0]):
        for j in range(down_sampled.shape[1]):
            down_sampled[i][j] = input_image[i * 2][j * 2]
    return down_sampled


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


filter_size = 3

image = cv.imread("../data/Barbara.bmp")

pixel_count = 0
gaussian = make_filter(filter_size)

level = 1
cv.imwrite("../output/Barbara_pyramid_level_{}.jpg".format(level), image)
pixel_count += (image.shape[0] * image.shape[1])
level += 1

new_image = image
while True:
    new_image = apply_filter(new_image, gaussian)
    new_image = down_sample_replication(new_image)
    cv.imwrite("../output/Barbara_pyramid_level_{}.jpg".format(level), new_image)
    level += 1
    pixel_count += (new_image.shape[0] * new_image.shape[1])
    if new_image.shape[0] == 1:
        break

print(level)
print(pixel_count)
print(image.shape[0] * image.shape[1])
