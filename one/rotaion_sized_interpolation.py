#  همانند فایل rotation_interpolation.py عمل میکند با این تفاوت که سایز تصویر نهایی را نیر محاسبه میکند تا گوشه‌‌های تصویر از بین نروند

import cv2 as cv
import numpy as np

image = cv.imread('../data/Elaine.jpg', cv.IMREAD_GRAYSCALE)
image_size = image.shape
# interpolation mod
# 0 -> nearest
# 1 -> bilinear
interpolation_mod = int(input("interpolation mode = "))
mods = ["nearest", "bilinear"]
thetas = [np.pi / 6, np.pi / 4, (np.pi / 2) - (np.pi / 9)]
thetas_degree = [30, 45, 80]
for theta_number in range(len(thetas)):
    theta = thetas[theta_number]
    transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    inv_transform = np.linalg.inv(transform)

    # چهار گوشه‌ی عکس را با استفاده از تبدیل مستقیم به نقاطی از تصویر خروجی مپ می‌کنیم
    corners_point = [[0, 0], [0, image_size[0]], [image_size[1], 0], [image_size[0], image_size[1]]]
    transformed_points = []
    for corner in corners_point:
        transformed_points.append(np.dot(corner, transform))

    # محاسبه‌ی ابعاد تصویر خروجی با استفاده از نقاط تبدیل یافته
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    for point in transformed_points:
        max_x = max(max_x, point[0])
        max_y = max(max_y, point[1])
        min_x = min(min_x, point[0])
        min_y = min(min_y, point[1])

    max_x = int(max_x)
    min_x = int(min_x)
    max_y = int(max_y)
    min_y = int(min_y)

    new_image_size = (max_x - min_x, max_y - min_y)
    new_image = np.zeros(new_image_size, dtype=np.uint8)
    count = 0
    count_2 = 0
    for i in range(int(-new_image_size[0] / 2), int(new_image_size[0] / 2)):
        for j in range(int(-new_image_size[1] / 2), int(new_image_size[1] / 2)):
            target = np.dot(np.array([i, j]), inv_transform)
            x = target[0] + 256
            y = target[1] + 256
            if x < 0 or y < 0 or x >= image_size[0] - 1 or y >= image_size[1] - 1:
                continue
            value = 2
            if interpolation_mod == 0:
                x = int(np.floor(x))
                y = int(np.floor(y))
                value = image[x][y]
            elif interpolation_mod == 1:
                if x == int(np.ceil(x)) or y == int(np.ceil(y)):
                    x = int(round(x))
                    y = int(round(y))
                    new_image[int(i + (new_image_size[0] / 2))][int(j + (new_image_size[0] / 2))] = image[x][y]
                    continue
                one = int(np.ceil(x))
                two = int(np.ceil(y))
                three = int(np.floor(x))
                four = int(np.floor(y))
                coe = np.array([[one, two, one * two, 1],
                                [one, four, one * four, 1],
                                [three, two, three * two, 1],
                                [three, four, three * two, 1]])
                b = np.array([image[one][two], image[one][four], image[three][two], image[three][four]])
                c1_c4 = np.dot(np.linalg.inv(coe), b)
                value = round((c1_c4[0] * x) + (c1_c4[1] * y) + (c1_c4[2] * x * y) + c1_c4[3])

            new_image[int(i + (new_image_size[0] / 2))][int(j + (new_image_size[0] / 2))] = value

    cv.imwrite(
        "../output/Elaine_rotate_{}_interpolation_{}.jpg".format(thetas_degree[theta_number], mods[interpolation_mod]),
        new_image)
