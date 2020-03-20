import numpy as np
from skimage import data
#im (3, H, W)
nbin = 50
im = data.astronaut()

color_histogram = []

for c in range(3):
    #histogram on color plane c
    histogram = np.zeros((nbin, ))
    for row in range(im.shape[1]):
        for col in range(im.shape[2]):
            pixel = im[c, row, col] #scalar
            bin = pixel * nbin // 256
            histogram[bin] +=1
    color_histogram = np.concatenate((color_histogram,histogram))

out = color_histogram/np.sum(color_histogram) #L1 norm
