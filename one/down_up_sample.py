import cv2 as cv
import numpy as np


def average_filter(input_image, filter_size):
    filter = np.ones([filter_size, filter_size])
    filter_sum = 0
    for row in filter:
        for data in row:
            filter_sum += data

    padded_image = np.pad(input_image, ((int(filter_size / 2), int(filter_size / 2)),
                                        (int(filter_size / 2), int(filter_size / 2))), mode="edge")
    filtered_image = np.zeros(input_image.shape, dtype=np.uint8)
    for i in range(padded_image.shape[0] - filter_size):
        for j in range(padded_image.shape[1] - filter_size):
            selection = []
            for h in range(filter_size):
                temp_row = []
                for k in range(filter_size):
                    temp_row.append(padded_image[i + h][j + k])
                selection.append(temp_row)
            selection = np.array(selection)
            filtered_image[i][j] = int(np.vdot(selection, filter) / filter_sum)
    return filtered_image


def down_sample_replication(input_image):
    down_sampled = np.zeros((int(input_image.shape[0] / 2), int(input_image.shape[1] / 2)), dtype=np.uint8)
    for i in range(down_sampled.shape[0]):
        for j in range(down_sampled.shape[1]):
            down_sampled[i][j] = input_image[i * 2][j * 2]
    return down_sampled


def up_sample_with_interpolation(input_image):
    global value
    output_image = np.zeros((input_image.shape[0] * 2, input_image.shape[1] * 2), dtype=np.uint8)
    transform = np.array([[2, 0], [0, 2]])
    inv_transform = np.linalg.inv(transform)

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            [x, y] = np.dot(np.array([i, j]), inv_transform)
            if 255 < x:
                x = 255
            if 255 < y:
                y = 255
            elif x == int(np.floor(x)) or y == int(np.floor(y)):
                x = int(round(x))
                y = int(round(y))
                value = input_image[x][y]
            else:
                one = int(np.ceil(x))
                two = int(np.ceil(y))
                three = int(np.floor(x))
                four = int(np.floor(y))
                coe = np.array([[one, two, one * two, 1],
                                [one, four, one * four, 1],
                                [three, two, three * two, 1],
                                [three, four, three * two, 1]])
                b = np.array(
                    [input_image[one][two], input_image[one][four], input_image[three][two], input_image[three][four]])
                c1_c4 = np.dot(np.linalg.inv(coe), b)
                value = round((c1_c4[0] * x) + (c1_c4[1] * y) + (c1_c4[2] * x * y) + c1_c4[3])
            output_image[i][j] = value
    return output_image


def up_sample_with_replication(input_image):
    output_image = np.zeros((input_image.shape[0] * 2, input_image.shape[1] * 2), dtype=np.uint8)
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i][j] = input_image[int(np.floor(i / 2))][int(np.floor(j / 2))]
    return output_image


image = cv.imread('../data/Goldhill.jpg', cv.IMREAD_GRAYSCALE)
average_image = average_filter(image, 3)

down_image = down_sample_replication(image)
down_average_image = down_sample_replication(average_image)

up_image = up_sample_with_interpolation(down_image)
up_replication_image = up_sample_with_replication(down_image)
up_average_image = up_sample_with_interpolation(down_average_image)
up_average_replication_image = up_sample_with_replication(down_average_image)

mse = [np.mean(np.square(np.subtract(image, up_image))),
       np.mean(np.square(np.subtract(image, up_replication_image))),
       np.mean(np.square(np.subtract(image, up_average_image))),
       np.mean(np.square(np.subtract(image, up_average_replication_image)))]

print("Pixel Replication     Bilinear interpolation")
print("Averaging ", mse[3], mse[2])
print("Remove Row&Column ", mse[1], mse[0])

cv.imwrite("../output/up.jpg", up_image)
cv.imwrite("../output/up_rep.jpg", up_replication_image)
cv.imwrite("../output/up_avg.jpg", up_average_image)
cv.imwrite("../output/up_avg_rep.jpg", up_average_replication_image)
