import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def save_filters(mag_spectrum):
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        im = plt.imshow(mag_spectrum[i], cmap='hot', interpolation="nearest", vmin=mag_spectrum[i].min(),
                        vmax=mag_spectrum[i].max())
        plt.colorbar(im)
        plt.title("filter_{}".format(i))
        plt.xticks([])
        plt.yticks([])

    plt.savefig("../output/filters_in_fourier.jpg")


image = cv.imread("../data/Barbara.jpg", cv.IMREAD_GRAYSCALE)
IMAGE_SIZE = image.shape
filter_a = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
filter_a_separate = np.array([1, 2, 1]) / 4
filter_b = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
filter_c = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
filters = [filter_a, filter_b, filter_c]

padding_width = [[254, 255], [254, 255]]
padded_filter = [np.pad(temp_filter, padding_width, constant_values=0) for temp_filter in filters]
fft_filters = [np.fft.fft2(x) for x in padded_filter]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]

save_filters(mag_spectrum)

fft_image = np.fft.fftshift(np.fft.fft2(image))
filtered_images = []
for temp_filter in mag_spectrum:
    output = np.zeros(fft_image.shape, dtype=np.complex128)
    for i in range(fft_image.shape[0]):
        for j in range(fft_image.shape[1]):
            output[i][j] = fft_image[i][j] * temp_filter[i][j]
    filtered_images.append(np.fft.ifft2(np.fft.ifftshift(output)))

output = np.zeros(fft_image.shape, dtype=np.complex128)
padded_filter_separate = np.fft.fftshift(np.fft.fft(np.pad(filter_a_separate, [254, 255], constant_values=0)))
padded_filter_bigger = np.zeros(fft_image.shape, dtype=np.complex128)
for i in range(fft_image.shape[0]):
    for j in range(fft_image.shape[1]):
        padded_filter_bigger[i][j] = padded_filter_separate[i] * padded_filter_separate[j]
padded_filter_separate = np.log(1 + np.abs(padded_filter_bigger))
for i in range(fft_image.shape[0]):
    for j in range(fft_image.shape[1]):
        output[i][j] = fft_image[i][j] * padded_filter_separate[i][j]
output = np.fft.ifft2(np.fft.ifftshift(output))

print(np.abs(output - filtered_images[0]).mean())

cv.imwrite("../output/Barbara_filtered_a.jpg", np.abs(filtered_images[0]))
cv.imwrite("../output/Barbara_filtered_b.jpg", np.abs(filtered_images[1]))
cv.imwrite("../output/Barbara_filtered_c.jpg", np.abs(filtered_images[2]))
cv.imwrite("../output/Barbara_filtered_a_separate.jpg", np.abs(output))
