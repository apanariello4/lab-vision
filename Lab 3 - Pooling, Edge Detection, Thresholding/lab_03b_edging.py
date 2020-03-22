import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import signal
import cv2
from sklearn.preprocessing import normalize

im = skimage.data.coins()
im = np.swapaxes(im, 0, 1) #(512, 512)
im = im[:128,:128]

S_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float)
S_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], dtype = np.float) #(3, 3)

G_x = signal.convolve2d(im, S_x, mode='same')
G_y = signal.convolve2d(im, S_y, mode='same')

magnitude = np.hypot(G_x, G_y)/1081 #max value is 1081, we normalized
v = (magnitude*255)

theta = np.arctan2(G_x, G_y) #this is [-pi,pi[, we need to normalize to [0,180[
h = theta + np.pi #[0,2pi[
h /= (2*np.pi) #[0,1[
h *= 180 #[0,180[

s = np.full(v.shape, 255, dtype=np.uint8)#value can be anything [0,255]

hsv = np.transpose([h, s, v], (1,2,0)).astype(np.uint8)

out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#plt.imshow(out, cmap='hsv')
#plt.show()

#-----------------------------------------#

canny = cv2.Canny(im,100,100)

plt.subplot(121),plt.imshow(out)
plt.title('Sobel Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Canny Image'), plt.xticks([]), plt.yticks([])

plt.show()