import random
import numpy as np
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = np.swapaxes(im, 0, 1)
val = random.randint(0, 255)

out = (im>val).astype(np.uint8)*255

plt.imshow(out, cmap='gray')
plt.show()