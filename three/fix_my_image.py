import cv2 as cv
from three.util import *

image = cv.imread("../data/my_image.jpg", cv.IMREAD_GRAYSCALE)

out = cv.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply(image)
out = median_filter(out, 3)
box_filter = np.ones([3, 3], dtype=np.uint8) / (3 * 3)
out = apply_filter(out, box_filter)

cv.imwrite("../output/my_image.jpg", out)
