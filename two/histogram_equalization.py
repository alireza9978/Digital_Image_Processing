import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from two.util import make_histeq
from two.util import make_histogram
from two.util import make_pdf_cdf

image = cv.imread('../data/Camera Man.bmp', cv.IMREAD_GRAYSCALE)
image_histogram = make_histogram(image)
image_pdf, image_cdf = make_pdf_cdf(image_histogram)
image_histeq = make_histeq(image, image_cdf, 255)
image_histeq_histogram = make_histogram(image_histeq)

cv_eq = cv.equalizeHist(image)
cv.imshow("cv_eq", cv_eq)
cv.imshow("mine_eq", image_histeq)
print(np.mean(cv_eq - image_histeq))
print(cv_eq)
print(image_histeq)
cv.waitKey(0)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].imshow(image, cmap="gray")
ax[0][1].bar(np.linspace(0, 255, 256), image_histogram)
ax[1][0].imshow(image_histeq, cmap="gray")
ax[1][1].bar(np.linspace(0, 255, 256), image_histeq_histogram)
plt.savefig("../output/two.jpg")
