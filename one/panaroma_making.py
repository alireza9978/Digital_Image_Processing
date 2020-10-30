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

(max_x, max_y, c) = car_two.shape
x, y = symbols('x y')
eq1 = Eq(c1_c4[0] * x + c1_c4[1] * y + c1_c4[2] * x * y + c1_c4[3], max_x)
eq2 = Eq(c5_c8[0] * x + c5_c8[1] * y + c5_c8[2] * x * y + c5_c8[3], max_y)
temp = solve((eq1, eq2), (x, y))

new_image_width = int(round(temp[0][0]))
new_image_height = int(round(temp[0][1]))
new_image = np.zeros((new_image_width + 1, new_image_height + 1, 3), dtype=np.uint8)
for i in range(new_image_width):
    for j in range(new_image_height):
        x = c1_c4[0] * i + c1_c4[1] * j + c1_c4[2] * i * j + c1_c4[3]
        y = c5_c8[0] * i + c5_c8[1] * j + c5_c8[2] * i * j + c5_c8[3]

        if not (x < 0 or y < 0 or x > car_two.shape[0] or y > car_two.shape[1]):
            new_image[i][j] = car_two[int(x)][int(y)]
        if -1 < i < 750 and -1 < j < 1000:
            new_image[i][j] = car_one[i][j]

cv.imshow("new_two", new_image)
cv.waitKey(0)
