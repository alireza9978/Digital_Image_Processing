import cv2 as cv
import numpy as np

image = cv.imread('../data/Building.jpg', 0)

dx, dy = np.gradient(image, 2)
Ixx = dx ** 2
Ixy = dy * dx
Iyy = dy ** 2

windows_size = 3
windows_size_half = int(np.floor(windows_size / 2))
k = 0.04
R = 20000000.00

Ixx = np.pad(Ixx, [windows_size_half, windows_size_half], mode="edge")
Ixy = np.pad(Ixy, [windows_size_half, windows_size_half], mode="edge")
Iyy = np.pad(Iyy, [windows_size_half, windows_size_half], mode="edge")

corner_list = []
r_matrix = []
for i in range(image.shape[0]):
    temp_r = []
    for j in range(image.shape[1]):
        end_i = i + windows_size
        end_j = j + windows_size

        Ixx_data = Ixx[i:end_i, j:end_j].sum()
        Ixy_data = Ixy[i:end_i, j:end_j].sum()
        Iyy_data = Iyy[i:end_i, j:end_j].sum()

        det = (Ixx_data * Iyy_data) - (Ixy_data ** 2)
        trace = Ixx_data + Iyy_data

        r = det - k * (trace ** 2)
        temp_r.append(r)
    r_matrix.append(temp_r)
r_matrix = np.array(r_matrix)
image = cv.imread('../data/Building.jpg')
output = np.zeros(image.shape)
output[r_matrix > R] = [255, 255, 255]
cv.imwrite("../output/building_corners.jpg", output)
