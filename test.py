import cv2 as cv
import numpy as np


img = cv.imread('img.png', cv.IMREAD_GRAYSCALE)

img1 = img.astype(np.float64)

img =img[:, :, :1]
if img is None:
    raise FileNotFoundError("Could not read 'img.png'")

linear_contrast = cv.convertScaleAbs(img, alpha=1.5, beta=0)
histogram_equalized = cv.equalizeHist(img)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
adaptive_equalized = clahe.apply(img)

cv.imshow('Original', img)
cv.imshow('Linear contrast', linear_contrast)
cv.imshow('Histogram equalization', histogram_equalized)
cv.imshow('CLAHE', adaptive_equalized)
cv.waitKey(0)
cv.destroyAllWindows()