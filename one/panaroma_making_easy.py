import cv2 as cv
import numpy as np

car_one = cv.imread('../data/Car1.jpg')
car_two = cv.imread('../data/Car2.jpg')
print(car_one.shape)
print(car_two.shape)
image_one = np.array([[462, 759], [313, 753], [389, 545], [399, 802],
                      [389, 941], [348, 457], [475, 940], [459, 560]])  # car one
image_two = np.array([[480, 341], [333, 337], [406, 122], [418, 383],
                      [409, 513], [361, 26], [490, 511], [479, 136]])  # car two

dif = np.array(np.round(np.mean(image_one - image_two, axis=0)))
new_car_avg = np.zeros((int(car_one.shape[0] - dif[0]), int(car_one.shape[1] + dif[1]), 3), dtype=np.uint8)
new_car = np.zeros((int(car_one.shape[0] - dif[0]), int(car_one.shape[1] + dif[1]), 3), dtype=np.uint8)

dif = np.array(dif, dtype=np.int)
for i in range(new_car.shape[0]):
    for j in range(new_car.shape[1]):
        i_one = i + dif[0]
        j_one = j
        i_two = i
        j_two = j - dif[1]

        value_one = None
        value_two = None
        if -1 < i_one < car_one.shape[0] and -1 < j_one < car_one.shape[1]:
            value_one = car_one[i_one][j_one]
        if -1 < i_two < car_two.shape[0] and -1 < j_two < car_two.shape[1]:
            value_two = car_two[i_two][j_two]

        if value_one is not None:
            new_car[i][j] = value_one
        if value_two is not None:
            new_car[i][j] = value_two

        if value_two is not None and value_one is not None:
            new_car_avg[i][j] = np.array(
                [int((int(value_one[0]) + int(value_two[0])) / 2),
                 int((int(value_one[1]) + int(value_two[1])) / 2),
                 int((int(value_one[2]) + int(value_two[2])) / 2)], dtype=np.uint8)
        elif value_one is not None:
            new_car_avg[i][j] = value_one
        elif value_two is not None:
            new_car_avg[i][j] = value_two

cv.imwrite("../output/panaroma_one.jpg", new_car)
cv.imwrite("../output/panaroma_one_avg.jpg", new_car_avg)

