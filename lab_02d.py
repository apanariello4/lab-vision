import random
import numpy as np
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
#im = np.swapaxes(im, 0, 1)

def histogram1c(im, nbin=256):
    histogram = np.zeros((nbin, ))
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            pixel = im[row, col] #scalar
            bin = pixel * nbin // 256
            histogram[bin] +=1
    return histogram


hist = histogram1c(im)
hist_cumsum = hist.cumsum() # cumulative hist

arr = np.arange(256)

thresh = -1
fn_min = np.inf

for t in range(1,255):
    p1,p2 = np.hsplit(hist,[t+1]) #prob

    w_1 = hist_cumsum[t+1]
    w_2 = hist_cumsum[255]-hist_cumsum[t+1]
    if w_1 < 1.e-6 or w_2 < 1.e-6:
        continue
    b1,b2 = np.hsplit(arr,[t+1])

    m_1 = np.sum(p1 * b1) / w_1
    m_2 = (np.sum(p2 * b2) / w_2)

    v1 = np.sum(((b1-m_1)**2)*p1)/w_1
    v2 = np.sum(((b2-m_2)**2)*p2)/w_2
    fn = v1 * w_1 + v2 * w_2
    if fn < fn_min:
        fn_min = fn
        thresh = t

ret2,th2 = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#with cv2

out = (im>thresh).astype(np.uint8)*255 #img with otsu thresh


#comparing manual otsu with cv2 otsu
plt.subplot(121),plt.imshow(out,cmap = 'gray')
plt.title(f'Manual {thresh}'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(th2,cmap = 'gray')
plt.title('cv2'), plt.xticks([]), plt.yticks([])

plt.show()
