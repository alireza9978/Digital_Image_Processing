import cv2 as cv
import numpy as np

image = cv.imread('../data/Elaine.jpg', cv.IMREAD_GRAYSCALE)
theta = -np.pi / 6
transform = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

image_array = []
x = 0
for row in image:
    y = 0
    for data in row:
        image_array.append(np.dot(transform, np.array([x - 256, y - 256, data])))
        y += 1
    x += 1

image_array = np.array(image_array)

new_image = np.zeros((512, 512), dtype=np.uint8)
for px in image_array:
    x = int(px[0] + 256)
    y = int(px[1] + 256)
    if -1 < x < 512 and -1 < y < 512:
        new_image[x][y] = px[2]

cv.imwrite("../output/Elaine_rotate_30_without_interpolation.jpg", new_image)

