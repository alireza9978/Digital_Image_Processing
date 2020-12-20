import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

girl_image = cv.imread("../data/Girl.bmp", cv.IMREAD_COLOR)
levels = [32, 16, 8]

girl_image_colors = np.reshape(girl_image, (girl_image.shape[0] * girl_image.shape[1], 3))

for level in levels:

    model = KMeans(level)
    model.fit(girl_image_colors)
    color_space = np.round(model.cluster_centers_).astype(np.uint8)
    output_colors = [color_space[i] for i in model.labels_]
    new_image = np.reshape(output_colors, girl_image.shape)

    cv.imwrite("../output/girl_with_{}_colors.jpg".format(level), new_image)
