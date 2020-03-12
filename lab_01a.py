import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data

a = 0.7328082888120535
b = 49.66048570066265
im = data.coffee()

#plt.imshow(im)
#plt.show()

out =  np.uint8(np.clip(np.round((im * a) + b),0,255))