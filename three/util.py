import random

import numpy as np


def apply_filter(image, input_filter):
    filter_size = input_filter.shape[0]
    padded_image = np.pad(image, int(filter_size / 2), mode='edge')
    output_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(padded_image.shape[0] - input_filter.shape[0]):
        for j in range(padded_image.shape[1] - input_filter.shape[1]):
            image_sample = np.take(np.take(padded_image, range(i, i + input_filter.shape[0]), axis=0),
                                   range(j, j + input_filter.shape[1]), axis=1)
            temp_sum = 0
            for h in range(input_filter.shape[0]):
                for k in range(input_filter.shape[1]):
                    temp_sum += (input_filter[h][k] * image_sample[h][k])
            output_image[i][j] = abs(int(temp_sum))
    return output_image


def apply_salt_and_pepper_noise(image, density):
    row_change_count = int(image.shape[1] * density)
    row_population = range(image.shape[1])
    output_image = np.copy(image)
    for row in output_image:
        rnd = random.choices(row_population, k=row_change_count)
        rnd_state = random.choices([True, False], k=row_change_count)
        for i in range(row_change_count):
            if rnd_state[i]:
                row[rnd[i]] = 255
            else:
                row[rnd[i]] = 0
    return output_image


def median_filter_padded(image, filter_size):
    filter_size_half = int(filter_size / 2)
    padded_image = np.pad(image, filter_size_half, mode='edge')
    output_image = np.zeros(image.shape, dtype=np.uint8)
    sorted_index_median = int((filter_size * filter_size) / 2)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_sample = np.take(np.take(padded_image, range(i, i + filter_size), axis=0),
                                   range(j, j + filter_size), axis=1)
            output_image[i][j] = sorted(image_sample.flatten())[sorted_index_median]
    return output_image


def median_filter(image, filter_size):
    output_image = np.zeros(image.shape, dtype=np.uint8)
    filter_size_half = int(filter_size / 2)
    sorted_index_median = int((filter_size * filter_size) / 2)
    for i in range(filter_size_half, image.shape[0] - filter_size_half):
        for j in range(filter_size_half, image.shape[1] - filter_size_half):
            image_sample = np.take(np.take(image, range(i - filter_size_half, i + filter_size_half + 1), axis=0),
                                   range(j - filter_size_half, j + filter_size_half + 1), axis=1)
            output_image[i][j] = sorted(image_sample.flatten())[sorted_index_median]
    return output_image
