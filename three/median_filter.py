import cv2 as cv
import numpy as np

from three.util import apply_salt_and_pepper_noise, median_filter

image = cv.imread("../data/Elaine.jpg", cv.IMREAD_GRAYSCALE)

noise_density = [0.05, 0.1, 0.2]
windows_size = [3, 5, 7, 9]
results = []
for density in noise_density:
    noised_image = apply_salt_and_pepper_noise(image, density)
    cv.imwrite("../output/Elaine_pepper&salt_{}.jpg".format(density), noised_image)
    this_noise_result = []
    for windows in windows_size:
        filtered_image = median_filter(noised_image, windows)
        cv.imwrite("../output/Elaine_pepper&salt_{}_filtered_size_{}.jpg".format(density, windows), filtered_image)
        this_noise_result.append(np.mean(np.square(np.subtract(image, filtered_image))))
    results.append(this_noise_result)

print(results)
