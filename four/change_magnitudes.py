import cv2 as cv
import numpy as np

image = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)
IMAGE_SIZE = image.shape

fft_image = np.fft.fftshift(np.fft.fft2(image))
T = [1 / 4, 1 / 8]
N = fft_image.shape[0]
outputs = []
for t in T:
    temps = [fft_image.copy(), fft_image.copy(), fft_image.copy(), fft_image.copy(), fft_image.copy()]
    for i in range(fft_image.shape[0]):
        for j in range(fft_image.shape[1]):
            if t * N < i < ((1 - t) * N) and t * N < j < ((1 - t) * N):
                temps[0][i][j] = 0
            if 0 <= i <= t * N and 0 <= j <= t * N:
                temps[1][i][j] = 0
            if 0 <= i <= t * N and ((1 - t) * N) <= j <= N - 1:
                temps[2][i][j] = 0
            if ((1 - t) * N) <= i <= N - 1 and 0 <= j <= t * N:
                temps[3][i][j] = 0
            if ((1 - t) * N) <= i <= N - 1 and ((1 - t) * N) <= j <= N - 1:
                temps[4][i][j] = 0
    outputs.append(temps)

for i in range(2):
    for j in range(5):
        cv.imwrite("../output/lena_filtered_{}_t_{}.jpg".format(j, i),
                   np.abs(np.fft.ifft2(np.fft.ifftshift(outputs[i][j]))))

# temp = np.abs(np.fft.ifft2(np.fft.ifftshift(outputs[0][2])) - np.fft.ifft2(np.fft.ifftshift(outputs[0][3])))
# cv.imwrite("../output/lena_filtered_diff.jpg", temp)
# temp = np.abs(np.fft.ifft2(np.fft.ifftshift(outputs[0][1])) - np.fft.ifft2(np.fft.ifftshift(outputs[0][2])))
# cv.imwrite("../output/lena_filtered_diff_2.jpg", temp)
# temp = np.abs(np.fft.ifft2(np.fft.ifftshift(outputs[0][0])) - np.fft.ifft2(np.fft.ifftshift(outputs[0][1])))
# cv.imwrite("../output/lena_filtered_diff_3.jpg", temp)
