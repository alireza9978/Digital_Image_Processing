import random

import cv2 as cv
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)


def apply_salt_and_pepper_noise(image, density):
    row_change_count = int(image.shape[1] * density)
    row_population = range(image.shape[1])
    output_image = np.copy(image)
    for row in output_image:
        rnd = random.choices(row_population, k=row_change_count)
        rnd_state = random.choices([True, False], k=row_change_count)
        for i in range(row_change_count):
            if rnd_state[i]:
                row[rnd[i]] = 255
            else:
                row[rnd[i]] = 0
    return output_image


image = cv.imread("../data/Lena.bmp")
gauss = cv.imread("../output/lena_gaussian_05.jpg")
pepper = apply_salt_and_pepper_noise(image, 0.10)

cv.imwrite("../output/add_pepper_salt_noise_Lena.jpg", pepper)
print("gaussian_noise_PSNR ", cv.PSNR(image, gauss))
print("pepper_noise_PSNR ", cv.PSNR(image, pepper))

images = [gauss, pepper]
images_names = ["gaussian_noise", "pepper_noise"]
result_name = ["BayesShrink", "VisuShrink", "VisuShrink2", "VisuShrink4"]
count = 0

for noisy in images:
    result = []
    # Estimate the average noise standard deviation across color channels.
    sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    # Due to clipping in random_noise, the estimate will be a bit smaller than the
    # specified sigma.
    print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

    im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, method='BayesShrink', mode='soft',
                               rescale_sigma=True)
    im_visushrink = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                    method='VisuShrink', mode='soft',
                                    sigma=sigma_est, rescale_sigma=True)

    # VisuShrink is designed to eliminate noise with high probability, but this
    # results in a visually over-smooth appearance.  Repeat, specifying a reduction
    # in the threshold by factors of 2 and 4.
    im_visushrink2 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                     method='VisuShrink', mode='soft',
                                     sigma=sigma_est / 2, rescale_sigma=True)
    im_visushrink4 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                     method='VisuShrink', mode='soft',
                                     sigma=sigma_est / 4, rescale_sigma=True)

    result.append(im_bayes)
    result.append(im_visushrink)
    result.append(im_visushrink2)
    result.append(im_visushrink4)
    for i in range(len(result)):
        result[i] = (result[i] * 255).astype(np.uint8)
        cv.imwrite("../output/" + result_name[i] + "_" + images_names[count] + ".jpg", result[i])
        print("PSNR for " + result_name[i] + "_" + images_names[count] + ".jpg " + str(cv.PSNR(image, result[i])))
    count += 1
