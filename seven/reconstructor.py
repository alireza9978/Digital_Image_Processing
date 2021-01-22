import random

import cv2 as cv
import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

target_images_one = ["../data/Attack 1/1.bmp",
                     "../data/Attack 1/2.bmp",
                     "../data/Attack 1/3.bmp",
                     "../data/Attack 1/4.bmp"]
target_images_two = ["../data/Attack 2/1.bmp",
                     "../data/Attack 2/2.bmp",
                     "../data/Attack 2/3.bmp",
                     "../data/Attack 2/4.bmp"]
image = cv.imread('../data/Reference.bmp')
main_images = cv.imread('../data/Original.bmp')
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
keyPoint, des = sift.detectAndCompute(image, None)
img = cv.drawKeypoints(image, keyPoint, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('../output/sift_key_points.jpg', img)
all_mp = []
all_mse = []
all_ssim = []
for i in range(len(target_images_one)):
    image_path = target_images_one[i]
    target_image = cv.imread(image_path)
    target_keyPoint, target_des = sift.detectAndCompute(target_image, None)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(des, target_des), key=lambda x: x.distance)
    print("matches count = ", len(matches))
    all_mp.append(len(matches))
    img3 = cv.drawMatches(image, keyPoint, target_image, target_keyPoint, matches[:20], target_image, flags=2)
    cv.imwrite('../output/matches_Attack_1_{}.jpg'.format(i + 1), img3)

    pts1 = [[keyPoint[x.queryIdx].pt[0], keyPoint[x.queryIdx].pt[1]] for x in matches]
    pts2 = [[target_keyPoint[x.trainIdx].pt[0], target_keyPoint[x.trainIdx].pt[1]] for x in matches]
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    k = 0
    best = None
    best_value = 0
    while k < 100:
        temp_points = random.choices(range(pts1.shape[0]), k=3)

        a = pts1[temp_points]
        b = pts2[temp_points]

        M = cv.getAffineTransform(b, a)
        image_path = target_images_two[i]
        target_image = cv.imread(image_path)
        converted = cv.warpAffine(target_image, M, (image.shape[0], image.shape[1]))
        mse_value = ssim(main_images, converted, multichannel=True)
        if mse_value > best_value:
            best = converted
            best_value = mse_value
        k += 1

    temp_mse = mse(main_images, best)
    temp_ssim = ssim(main_images, best, multichannel=True)
    all_mse.append(temp_mse)
    all_ssim.append(temp_ssim)
    print("MSE = ", temp_mse)
    print("SSIM = ", temp_ssim)
    cv.imwrite("../output/converted_Attack_2_{}.jpg".format(i + 1), best)

all_mp = np.array(all_mp)
all_mse = np.array(all_mse)
all_ssim = np.array(all_ssim)
print("mean MP = ", all_mp.mean())
print("mean MSE = ", all_mse.mean())
print("mean SSIM = ", all_ssim.mean())
print("std MP = ", all_mp.std())
print("std MSE = ", all_mse.std())
print("std SSIM = ", all_ssim.std())
