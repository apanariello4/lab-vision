from io import BytesIO
import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt


with open("gallery_0.jpg", "rb") as gallery_0:
    bytes = np.asarray(bytearray(gallery_0.read()), dtype=np.uint8)
im_a = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
im_a = np.swapaxes(np.swapaxes(im_a, 0, 2), 1, 2)
im_a = im_a[::-1, :, :]  # from BGR to RGB

with open("gallery_1.jpg", "rb") as gallery_1:
    bytes = np.asarray(bytearray(gallery_1.read()), dtype=np.uint8)
im_b = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
im_b = np.swapaxes(np.swapaxes(im_b, 0, 2), 1, 2)
im_b = im_b[::-1, :, :]  # from BGR to RGB
#############################################

im_a = np.transpose(im_a,(1,2,0))
im_b = np.transpose(im_b,(1,2,0))
im_a = im_a[:-1,:,:]

gal0 = np.float32([[195,34],[181,237],[111,210],[311,209]])
gal1 = np.float32([[139,51],[133,190],[94,165],[331,196]])

rect_in = np.array([[195,40],[183,232],[111,210],[308,207],[314,96]], dtype="float32")
rect_out = np.array([[144,55],[137,190],[94,165],[331,194],[336,62]], dtype="float32")

M, _ = cv2.findHomography(rect_in,rect_out,cv2.RANSAC)
iM = np.linalg.inv(M)

height, width = im_a.shape[:2]

dst = cv2.warpPerspective(im_b, M, (width,height))

mask = np.all(dst == [0, 0, 0], axis=-1, keepdims=True) #(286, 509)

dst[mask] = im_a[mask]
#out = cv2.add(dst,im_a,mask=np.floatmask)

plt.imshow(dst)
plt.show()


