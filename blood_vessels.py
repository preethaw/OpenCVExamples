#detects blood vessels of a fundus image

import cv2;
import numpy as np;
from pylab import array, uint8,arange;
from matplotlib import pyplot as plt

img = cv2.imread("images\\01_dr.jpg");
input = cv2.imread("images\\01_dr.jpg");

green = img[:,:,1]
g_inv = cv2.bitwise_not(green)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(g_inv)


kernel = np.ones((51,51),np.uint8)
kernel_small = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(cl_img, cv2.MORPH_OPEN, cv2.getStructuringElement( cv2.MORPH_ELLIPSE,(91,91)))

godisk = cv2.subtract(cl_img,opening)
kernel_tiny = np.ones((2,2),np.uint8)
erode1  = cv2.erode(godisk,kernel_small)
dilate1  = cv2.dilate(erode1,kernel_tiny)
med_img  = cv2.medianBlur(godisk,13);

kernel2 = np.ones((91,91),np.uint8)
background = cv2.morphologyEx(med_img, cv2.MORPH_OPEN, kernel2)
I2 = cv2.subtract(med_img,background)

# equalize the histogram of the Y channel
eq_I2 = cv2.equalizeHist(I2)

ret1,thr = cv2.threshold(eq_I2,238,255,cv2.THRESH_BINARY)

kernel = np.ones((3,3), 'uint8')
dil = cv2.dilate(thr, kernel)  
cv2.namedWindow("BV",cv2.WINDOW_NORMAL);
cv2.imshow("BV",dil)

im2, contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
mask = np.ones(dil.shape[:2], dtype="uint8") * 255

connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(dil, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]

area1 = stats[labels, cv2.CC_STAT_AREA]
print(area1);
print(len(output))

for cnt in contours:
    area = cv2.arcLength(cnt,True)
    
    if area < 50:
        cv2.drawContours(mask, [cnt], -1, 0, -1)

# remove the contours from the image and show the resulting images
dil = cv2.bitwise_and(dil, dil, mask=mask)

cv2.drawContours(img,contours,-1,(0,255,0),3)
cv2.namedWindow("Blood_Vessels_overlapped",cv2.WINDOW_NORMAL);
cv2.imshow("Blood_Vessels_overlapped",img)
cv2.waitKey(0);

