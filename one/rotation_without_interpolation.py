import cv2 as cv
import numpy as np

image = cv.imread('../data/Elaine.jpg', cv.IMREAD_GRAYSCALE)
teta = -np.pi / 6
transform = np.array([[np.cos(teta), -np.sin(teta), 0], [np.sin(teta), np.cos(teta), 0], [0, 0, 1]])

image_array = []
x = 0
for row in image:
    y = 0
    for data in row:
        image_array.append(np.dot(transform, np.array([x - 256, y - 256, data])))
        y += 1
    x += 1

image_array = np.array(image_array)
# max_x = image_array.max(axis=0, initial=0)
# max_y = image_array.max(axis=1, initial=0)
# min_x = image_array.min(axis=0, initial=512)
# min_y = image_array.min(axis=1, initial=512)

new_image = np.zeros((512, 512), dtype=np.uint8)
for px in image_array:
    x = int(px[0] + 256)
    y = int(px[1] + 256)
    if -1 < x < 512 and -1 < y < 512:
        new_image[x][y] = px[2]

cv.imshow("image", new_image)
cv.waitKey(0)
