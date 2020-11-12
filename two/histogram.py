import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from two.util import make_histogram

image = cv.imread('../data/Camera Man.bmp', cv.IMREAD_GRAYSCALE)
image_histogram = make_histogram(image)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(image, cmap="gray")
ax[1].bar(np.linspace(0, 255, 256), image_histogram)
plt.savefig("../output/one.jpg")
