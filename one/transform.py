import cv2 as cv
import numpy as np


def get_four_control_point(theta, picture_size):
    # we calculate new position for 4 point
    start_angel = [np.pi / 4, -np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4]
    end_angel = [start - theta for start in start_angel]
    control_point = []
    start_point = []
    diagonal_side_length = np.sqrt(((picture_size[0] / 2) ** 2) + ((picture_size[1] / 2) ** 2))
    for i in range(4):
        angel = end_angel[i]
        end_v = np.cos(angel) * diagonal_side_length
        end_w = np.sin(angel) * diagonal_side_length
        angel = start_angel[i]
        start_x = np.cos(angel) * diagonal_side_length
        start_y = np.sin(angel) * diagonal_side_length

        control_point.append([end_v, end_w])
        start_point.append([start_x, start_y])
    return control_point, start_point


image = cv.imread('../data/Elaine.jpg', cv.IMREAD_GRAYSCALE)
image_size = image
theta = np.pi / 4
transform30 = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

tie, xy = get_four_control_point(theta, (512, 512))
coe = np.array([[tie[0][0], tie[0][1], tie[0][0] * tie[0][1], 1],
                [tie[1][0], tie[1][1], tie[1][0] * tie[1][1], 1],
                [tie[2][0], tie[2][1], tie[2][0] * tie[2][1], 1],
                [tie[3][0], tie[3][1], tie[3][0] * tie[3][1], 1]])
b = np.array([xy[0][0], xy[1][0], xy[2][0], xy[3][0]])
c = np.array([xy[0][1], xy[1][1], xy[2][1], xy[3][1]])
c1_c4 = np.linalg.inv(coe).dot(b)
c5_c8 = np.linalg.inv(coe).dot(c)

max_x = 0
min_x = 0
max_y = 0
min_y = 0
for i in range(4):
    max_x = max(max_x, tie[i][0])
    max_y = max(max_y, tie[i][1])
    min_x = min(min_x, tie[i][0])
    min_y = min(min_y, tie[i][1])

new_image_width = int(max_x - min_x)
new_image_height = int(max_y - min_y)
new_image = np.zeros((new_image_width + 1, new_image_height + 1), dtype=np.uint8)

for i in range(int(-new_image_width / 2), int(new_image_width / 2)):
    for j in range(int(-new_image_height / 2), int(new_image_height / 2)):
        x = c1_c4[0] * i + c1_c4[1] * j + c1_c4[2] * i * j + c1_c4[3]
        y = c5_c8[0] * i + c5_c8[1] * j + c5_c8[2] * i * j + c5_c8[3]
        x = round(x + 256)
        y = round(y + 256)
        if x < 0 or y < 0 or x > 511 or y > 511:
            continue
        new_image[int(i + (new_image_width / 2))][int(j + (new_image_height / 2))] = image[int(x)][int(y)]

cv.imshow("image", new_image)
cv.waitKey(0)
