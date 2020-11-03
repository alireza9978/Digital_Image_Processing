import cv2 as cv
import numpy as np

# گرفتن ورودی برای روش تبدیل عکس و خواندن تصویر
image = cv.imread('../data/Elaine.jpg', cv.IMREAD_GRAYSCALE)
image_size = image.shape
# interpolation mod
# 0 -> nearest
# 1 -> bilinear
interpolation_mod = int(input("interpolation mode = "))
mods = ["nearest", "bilinear"]
thetas = [np.pi / 6, np.pi / 4, (np.pi / 2) - (np.pi / 9)]
thetas_degree = [30, 45, 80]
# محاسبه‌ی خروجی برای هر کدام از زاویه‌ها
for theta_number in range(len(thetas)):
    theta = thetas[theta_number]
    # محاسبه‌ی تبدیل مستقیم و معکوس
    transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    inv_transform = np.linalg.inv(transform)

    new_image = np.zeros(image_size, dtype=np.uint8)
    count = 0
    count_2 = 0
    # برای این که تصویر حول مرکز دوران کند مرکز ۰ و ۰ در نظر گرفته شده است
    for i in range(int(-image_size[0] / 2), int(image_size[0] / 2)):
        for j in range(int(-image_size[1] / 2), int(image_size[1] / 2)):
            # محاسبه‌ی نقطه‌ای از تصویر مبدا برای هر نقطه‌ی تصویر نهایی
            target = np.dot(np.array([i, j]), inv_transform)
            x = target[0] + 256
            y = target[1] + 256

            if x < 0 or y < 0 or x > 511 or y > 511:
                continue
            value = 2
            if interpolation_mod == 0:
                # انتخاب نزدیک ترین پیکسل از تصویر مبدا به نقطه‌ی بدست امده
                x = int(round(x))
                y = int(round(y))
                value = image[x][y]
            elif interpolation_mod == 1:
                if x == int(np.ceil(x)) or y == int(np.ceil(y)):
                    x = int(round(x))
                    y = int(round(y))
                    new_image[int(i + (image_size[0] / 2))][int(j + (image_size[0] / 2))] = image[x][y]
                    continue
                # محاسبه‌ی چهار پیکسل اطراف نقطه‌ی بدست آمده
                #  و محاسبه‌ی مقدار با استفاده از میانگین آن چهار نقطه
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

            new_image[int(i + (image_size[0] / 2))][int(j + (image_size[0] / 2))] = value

    # ذخیره کردن هر تصویر با نام مشخص
    cv.imwrite("../output/Elaine_rotate_{}_interpolation_{}.jpg".format(thetas_degree[theta_number], mods[interpolation_mod]),
               new_image)
