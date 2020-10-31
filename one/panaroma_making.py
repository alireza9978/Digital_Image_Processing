import cv2 as cv
import numpy as np
from sympy import symbols, Eq, solve

car_one = cv.imread('../data/Car1.jpg')
car_two = cv.imread('../data/Car2.jpg')

image_one = np.array([[462, 759], [313, 753], [389, 545], [399, 802],
                      [389, 941], [348, 457], [475, 940], [459, 560]])  # car one
image_two = np.array([[480, 341], [333, 337], [406, 122], [418, 383],
                      [409, 513], [361, 26], [490, 511], [479, 136]])  # car two
coe = []
b = []
c = []
for i in range(4, 8):
    coe.append([image_one[i][0], image_one[i][1], image_one[i][0] * image_one[i][1], 1])
    b.append(image_two[i][0])
    c.append(image_two[i][1])
coe = np.array(coe)
b = np.array(b)
c = np.array(c)
c1_c4 = np.linalg.inv(coe).dot(b)
c5_c8 = np.linalg.inv(coe).dot(c)

image_two_corners = [[0, 0], [0, car_two.shape[0]], [car_two.shape[1], 0], [car_two.shape[0], car_two.shape[1]]]
image_two_transformed_corners = []
for point in image_two_corners:
    x, y = symbols('x y')
    eq1 = Eq(c1_c4[0] * x + c1_c4[1] * y + c1_c4[2] * x * y + c1_c4[3], point[0])
    eq2 = Eq(c5_c8[0] * x + c5_c8[1] * y + c5_c8[2] * x * y + c5_c8[3], point[1])
    temp = solve((eq1, eq2), (x, y))
    image_two_transformed_corners.append(temp[0])

max_x = 0
min_x = 0
max_y = 0
min_y = 0
for point in image_two_transformed_corners:
    max_x = max(max_x, point[0])
    max_y = max(max_y, point[1])
    min_x = min(min_x, point[0])
    min_y = min(min_y, point[1])

max_x = int(max_x)
min_x = int(min_x)
max_y = int(max_y)
min_y = int(min_y)

car_two_transformed_width = int(np.floor(max_x - min_x))
car_two_transformed_height = int(np.floor(max_y - min_y))
car_two_transformed = np.zeros((car_two_transformed_width + 2, car_two_transformed_height + 2, 3), dtype=np.uint8)
for i in range(min_x, max_x):
    for j in range(min_y, max_y):
        x = c1_c4[0] * i + c1_c4[1] * j + c1_c4[2] * i * j + c1_c4[3]
        y = c5_c8[0] * i + c5_c8[1] * j + c5_c8[2] * i * j + c5_c8[3]
        if not (x < 0 or y < 0 or x >= car_two.shape[0] or y >= car_two.shape[1]):
            car_two_transformed[i - min_x][j - min_y] = car_two[int(x)][int(y)]

new_image_width = int(round(max_x))
new_image_height = int(round(max_y))
new_image = np.zeros((new_image_width + 1, new_image_height + 1, 3), dtype=np.uint8)
for i in range(new_image_width):
    for j in range(new_image_height):
        x = c1_c4[0] * i + c1_c4[1] * j + c1_c4[2] * i * j + c1_c4[3]
        y = c5_c8[0] * i + c5_c8[1] * j + c5_c8[2] * i * j + c5_c8[3]

        if not (x < 0 or y < 0 or x > car_two.shape[0] or y > car_two.shape[1]):
            new_image[i][j] = car_two[int(x)][int(y)]
        if -1 < i < 750 and -1 < j < 1000:
            new_image[i][j] = car_one[i][j]

cv.imwrite("../output/car_two_transformed.jpg", car_two_transformed)
cv.imwrite("../output/panaroma_two.jpg", new_image)
