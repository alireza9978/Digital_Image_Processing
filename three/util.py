import numpy as np


def apply_filter(image, input_filter):
    filter_sum = 0
    for row in input_filter:
        for data in row:
            filter_sum += data
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
            output_image[i][j] = int(temp_sum/filter_sum)
    return output_image


