import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('images/01_dr.JPG',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
 
#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
titles = ['Original Image','Optic Disk',]
images = [img, thresh1]
 
for i in xrange(2):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
 
plt.show()
