import cv2 as cv
import numpy as np

image = cv.imread('../data/Elaine.jpg', cv.IMREAD_GRAYSCALE)
image_size = image.shape
theta = np.pi / 4

# interpolation mod
# 0 -> nearest
# 1 -> bilinear
interpolation_mod = int(input("interpolation mode = "))

transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
inv_transform = np.linalg.inv(transform)

new_image = np.zeros(image_size, dtype=np.uint8)
count = 0
count_2 = 0
for i in range(int(-image_size[0] / 2), int(image_size[0] / 2)):
    for j in range(int(-image_size[1] / 2), int(image_size[1] / 2)):
        target = np.dot(np.array([i, j]), inv_transform)
        x = target[0] + 256
        y = target[1] + 256
        if x < 0 or y < 0 or x > 511 or y > 511:
            continue
        value = 2
        if interpolation_mod == 0:
            x = int(round(x))
            y = int(round(y))
            value = image[x][y]
        elif interpolation_mod == 1:
            if x == int(np.ceil(x)) or y == int(np.ceil(y)):
                x = int(round(x))
                y = int(round(y))
                value = image[x][y]
                continue
            one = int(np.ceil(x))
            two = int(np.ceil(y))
            three = int(np.floor(x))
            four = int(np.floor(y))
            coe = np.array([[one, two, one * two, 1],
                            [one, four, one * four, 1],
                            [three, two, three * two, 1],
                            [three, four, three * two, 1]])
            b = np.array([image[one][two], image[one][four], image[three][two], image[three][four]])
            c1_c4 = np.dot(np.linalg.inv(coe), b)
            value = round((c1_c4[0] * x) + (c1_c4[1] * y) + (c1_c4[2] * x * y) + c1_c4[3])

        new_image[int(i + (image_size[0] / 2))][int(j + (image_size[0] / 2))] = value

cv.imshow("image", new_image)
cv.waitKey(0)
