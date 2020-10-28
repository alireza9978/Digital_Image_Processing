import cv2 as cv
import numpy as np

image = cv.imread('../data/Elaine.jpg', cv.IMREAD_GRAYSCALE)
image_size = image.shape
teta = np.pi / 4

# interpolation mod
# 0 -> nearest
# 1 -> bilinear
# 2 -> cubdric
interpolation_mod = 0

transform = np.array([[np.cos(teta), -np.sin(teta)], [np.sin(teta), np.cos(teta)]])
inv_transform = np.linalg.inv(transform)

new_image = np.zeros(image_size, dtype=np.uint8)

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
            value = 1
        new_image[int(i + (image_size[0] / 2))][int(j + (image_size[0] / 2))] = value

cv.imshow("image", new_image)
cv.waitKey(0)
