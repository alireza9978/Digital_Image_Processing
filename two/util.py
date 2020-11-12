import numpy as np


# تابعی برای ساختن histogram
def make_histogram(input_image):
    # ساختن ارایه‌ای از صفر ها به طول تعداد سطوح خاکستری
    histogram = np.zeros(256)
    # شمارش تعداد سطوح خاکستری
    for row in input_image:
        for data in row:
            histogram[data] += 1
    return histogram


# تابعی برای ساختن pdf, cdf
def make_pdf_cdf(histogram):
    #  محاسبه تعداد کل پیکسل ها برای pdf
    total_count = np.sum(histogram)
    pdf = histogram / total_count
    # تبدیل pdf به cdf
    cdf = np.zeros(256)
    temp = 0
    for i in range(256):
        temp += pdf[i]
        cdf[i] = temp
    return pdf, cdf


# نرمال کردن histogram هر تصویر ورودی
def make_histeq(input_image, cdf, max_value):
    # محاسبه‌ی تابع تبدیل با استفاده از تعداد سطوح خاکستری نهایی و cdf
    temp_cdf = np.round(cdf * max_value)
    # ایجاد عکس نهایی و مقدار دهی هر پیکسل
    normal_image = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            normal_image[i][j] = temp_cdf[input_image[i][j]]
    return normal_image


def local_histogram_equalization(image, filter_ratio):
    new_image = np.zeros(image.shape, dtype=np.uint8)
    step = int(image.shape[0] / filter_ratio)
    if image.shape[0] % step != 0:
        return
    for i in range(0, image.shape[0], step):
        for j in range(0, image.shape[1], step):
            image_part = np.zeros((step, step), dtype=np.uint8)
            for k in range(0, step):
                for h in range(0, step):
                    image_part[k][h] = image[i + k][j + h]
            image_part_histogram = make_histogram(image_part)
            image_part_pdf, image_part_cdf = make_pdf_cdf(image_part_histogram)
            # image_part_histeq = make_histeq(image_part, image_part_cdf, image_part.max())
            image_part_histeq = make_histeq(image_part, image_part_cdf, 255)
            for k in range(0, step):
                for h in range(0, step):
                    new_image[i + k][j + h] = image_part_histeq[k][h]
    return new_image


def histogram_local(image, start_x, start_y, filter_size, new_image):
    image_part = np.zeros((filter_size, filter_size), dtype=np.uint8)
    for k in range(0, filter_size):
        for h in range(0, filter_size):
            test = image[start_x + k][start_y + h]
            image_part[k][h] = test

    image_part_histogram = make_histogram(image_part)
    image_part_pdf, image_part_cdf = make_pdf_cdf(image_part_histogram)
    image_part_histeq = make_histeq(image_part, image_part_cdf, 255)
    for k in range(0, filter_size):
        for h in range(0, filter_size):
            new_image[start_x + k][start_y + h] = image_part_histeq[k][h]


def local_histogram_equalization_two(image, filter_size):
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(0, image.shape[0] - filter_size, filter_size):
        for j in range(0, image.shape[1] - filter_size, filter_size):
            histogram_local(image, i, j, filter_size, new_image)
    for i in range(0, image.shape[0] - filter_size, filter_size):
        histogram_local(image, i, image.shape[1] - filter_size, filter_size, new_image)
    for j in range(0, image.shape[1] - filter_size, filter_size):
        histogram_local(image, image.shape[0] - filter_size, j, filter_size, new_image)
    histogram_local(image, image.shape[0] - filter_size, image.shape[1] - filter_size, filter_size, new_image)
    return new_image


def avg_histogram(histogram):
    my_avg_filter = np.ones(9)
    my_avg_filter_sum = 9
    padded_histogram = np.pad(histogram, pad_width=4, mode="edge")
    new_histogram = np.zeros(histogram.shape)
    for i in range(0, 255):
        temp = 0
        for j in range(9):
            temp += padded_histogram[i + j] * my_avg_filter[j]
        temp /= my_avg_filter_sum
        new_histogram[i] = temp
    return new_histogram


def get_slop(array):
    slop = np.zeros(255)
    for i in range(0, 255):
        slop[i] = array[i] - array[i + 1]
    return np.min(slop)


def get_cdf(histogram):
    total_count = np.sum(histogram)
    pdf = histogram / total_count
    # تبدیل pdf به cdf
    cdf = np.zeros(256)
    temp = 0
    for i in range(256):
        temp += pdf[i]
        cdf[i] = temp
    return cdf


# تابعی برای ساختن pdf, cdf
def make_pdf_cdf_limited(histogram):
    cdf = get_cdf(histogram)
    while get_slop(cdf) < -0.015:
        one_max = np.max(histogram) / 2
        for i in range(len(histogram)):
            if histogram[i] > one_max:
                histogram[i] = one_max
            else:
                histogram[i] += 1
        cdf = get_cdf(histogram)

    total_count = np.sum(histogram)
    pdf = histogram / total_count
    return pdf, cdf


def histogram_local_limited(image, start_x, start_y, filter_size, new_image):
    image_part = np.zeros((filter_size, filter_size), dtype=np.uint8)
    for k in range(0, filter_size):
        for h in range(0, filter_size):
            image_part[k][h] = image[start_x + k][start_y + h]

    image_part_histogram = make_histogram(image_part)
    image_part_pdf, image_part_cdf = make_pdf_cdf_limited(image_part_histogram)
    image_part_histeq = make_histeq(image_part, image_part_cdf, 255)
    for k in range(0, filter_size):
        for h in range(0, filter_size):
            new_image[start_x + k][start_y + h] = image_part_histeq[k][h]


def local_histogram_equalization_three(image, filter_size):
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(0, image.shape[0] - filter_size, filter_size):
        for j in range(0, image.shape[1] - filter_size, filter_size):
            histogram_local_limited(image, i, j, filter_size, new_image)
    for i in range(0, image.shape[0] - filter_size, filter_size):
        histogram_local_limited(image, i, image.shape[1] - filter_size, filter_size, new_image)
    for j in range(0, image.shape[1] - filter_size, filter_size):
        histogram_local_limited(image, image.shape[0] - filter_size, j, filter_size, new_image)
    histogram_local_limited(image, image.shape[0] - filter_size, image.shape[1] - filter_size, filter_size, new_image)
    return new_image
