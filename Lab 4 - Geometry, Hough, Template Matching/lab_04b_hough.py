from io import BytesIO
import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt

im = data.coins()[160:230, 70:270]
#############################################

cimg = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)

canny = cv2.Canny(im,150,255)

hough = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 30, param1=100,param2=30,minRadius=10,maxRadius=0)

circles = np.around(hough)
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(cimg, cmap='gray'),plt.show()