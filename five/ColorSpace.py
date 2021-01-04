import cv2 as cv
import numpy as np

pepper_image = cv.imread("../data/Pepper.bmp", cv.IMREAD_COLOR)
R = pepper_image[:, :, 2]
G = pepper_image[:, :, 1]
B = pepper_image[:, :, 0]

cv.imwrite("../output/pepper_red_to_gray.jpg", R)
cv.imwrite("../output/pepper_green_to_gray.jpg", G)
cv.imwrite("../output/pepper_blue_to_gray.jpg", B)

R = R.astype(np.float32)
G = G.astype(np.float32)
B = B.astype(np.float32)

H = []
S = []
I = []
for i in range(pepper_image.shape[0]):
    h_row = []
    s_row = []
    i_row = []
    for j in range(pepper_image.shape[1]):
        bot = np.power(R[i, j] - G[i, j], 2) + ((R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
        theta = 0
        if bot > 0:
            temp = (0.5 * (R[i, j] - G[i, j] + R[i, j] - B[i, j])) / np.sqrt(bot)
            theta = np.arccos(temp) * 180 / np.pi

        h = theta if B[i, j] <= G[i, j] else 360 - theta
        bot = (R[i, j] + G[i, j] + B[i, j])
        s = 0
        if bot != 0:
            s = 1 - ((3 * min(R[i, j], G[i, j], B[i, j])) / bot)
        hsi_i = (1 / 3) * (R[i, j] + G[i, j] + B[i, j])
        h_row.append(h)
        s_row.append(s)
        i_row.append(hsi_i)
    H.append(h_row)
    S.append(s_row)
    I.append(i_row)

H = np.array(H, dtype=np.float32)
S = np.array(S, dtype=np.float32)
I = np.array(I, dtype=np.float32)

H = (H / H.max()) * 255
S = (S / S.max()) * 255
I = (I / I.max()) * 255

cv.imwrite("../output/pepper_H.jpeg", H)
cv.imwrite("../output/pepper_S.jpeg", S)
cv.imwrite("../output/pepper_I.jpeg", I)
