# import the necessary packages
import numpy as np
import argparse
import cv2

# load the image and convert it to grayscale
image = cv2.imread("images/01_dr.JPG")

orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# the area of the image with the largest intensity value
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)

cv2.namedWindow("Naive", cv2.WINDOW_NORMAL) 
# display the results of the naive attempt
cv2.imshow("Naive", image)

# apply a Gaussian blur to the image then find the brightest
# region
gray = cv2.GaussianBlur(gray, (141, 141), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
image = orig.copy()
cv2.circle(image, maxLoc, 141, (255, 0, 0), 2)
cv2.circle(image,maxLoc,1,(0,0,255),3)
cv2.namedWindow("Robust", cv2.WINDOW_NORMAL) 
# display the results of our newly improved method
cv2.imshow("Robust", image)
cv2.waitKey(0)
