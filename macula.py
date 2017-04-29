import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('images/01_dr.JPG')


def split_green_channel(image):
  green = image[:,:,2]
  return green

def main():
    img = cv2.imread('images/01_dr.JPG')
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    cv2.imshow("Input",img)
    cv2.waitKey(0)

    red = split_green_channel(img)
    img = np.zeros((red.shape[0], red.shape[1], 3), dtype = red.dtype) 
    img[:,:,2] = red
    cv2.namedWindow("Input with Red filter", cv2.WINDOW_NORMAL)
    cv2.imshow("Input with Red filter",img)
    cv2.waitKey(0)
    return red

def detect_dark_areas(img):
    ret,dark_img = cv2.threshold(img,200,250,cv2.THRESH_BINARY)
    cv2.namedWindow("dark areas", cv2.WINDOW_NORMAL)
    cv2.imshow("dark areas",dark_img)
    im2, contours = cv2.findContours(dark_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ctr = np.array(contours).reshape((-1,1,2)).astype(np.int32)
    #cv2.drawContours(dark_img, [ctr], 0, (0, 255, 0), -1)
    cv2.drawContours(dark_img, [ctr], -1, (0,255,0), 2)
    #cv2.drawContours(dark_img, [ctr], 3, (0,255,0, 3)
    cv2.namedWindow("dark areas painted", cv2.WINDOW_NORMAL)
    cv2.imshow("dark areas painted",dark_img)
    cv2.waitKey(0)
    


if __name__ == "__main__":
    img = cv2.imread('images/01_dr.JPG')
    red = main()
    detect_dark_areas(red)
    



