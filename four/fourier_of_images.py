import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

lena = cv.imread("../data/Lena.bmp", cv.IMREAD_GRAYSCALE)
f16 = cv.imread("../data/F16.bmp", cv.IMREAD_GRAYSCALE)
baboon = cv.imread("../data/Baboon.bmp", cv.IMREAD_GRAYSCALE)

images = [lena, f16, baboon]
images_name = ["lena", "f16", "baboon"]
fft_images = [np.fft.fft2(image) for image in images]
fft_shifted_images = [np.fft.fftshift(image) for image in fft_images]
fft_scaled_images = [np.log(1 + np.abs(image)) for image in fft_images]
fft_scaled_shifted_images = [np.log(1 + np.abs(image)) for image in fft_shifted_images]

for i in range(3):
    plt.figure()
    plt.imshow(np.abs(fft_images[i]), cmap="gray")
    plt.colorbar()
    plt.savefig("../output/normal_fft_{}.jpg".format(images_name[i]))

for i in range(3):
    plt.figure()
    plt.imshow(np.abs(fft_shifted_images[i]), cmap="gray")
    plt.colorbar()
    plt.savefig("../output/shifted_fft_{}.jpg".format(images_name[i]))

for i in range(3):
    plt.figure()
    plt.imshow(np.abs(fft_scaled_images[i]), cmap="gray")
    plt.colorbar()
    plt.savefig("../output/scaled_fft_{}.jpg".format(images_name[i]))

for i in range(3):
    plt.figure()
    plt.imshow(np.abs(fft_scaled_shifted_images[i]), cmap="gray")
    plt.colorbar()
    plt.savefig("../output/scaled_shifted_fft_{}.jpg".format(images_name[i]))
