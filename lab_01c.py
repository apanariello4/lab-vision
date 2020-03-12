import numpy as np

n, iC, H, W = input.shape
oC, _, kH, kW = kernel.shape
out = np.zeros((n,oC,H-(kH-1),W-(kW-1)))
for n_img in range(n):
    for c in range(oC):
        for h in range(H-(kH-1)):
            for w in range(W-(kW-1)):
                out[n_img,c,h,w] = np.sum((input[n_img,:,h:h+kH,w:w+kW]*kernel[c,:,:,:]))