import random
import numpy as np

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = np.random.rand(n, iC, H, W)
kernel = np.random.rand(oC, iC, kH, kW)

n, iC, H, W = input.shape
oC, _, kH, kW = kernel.shape
out = np.zeros((n,oC,H-(kH-1),W-(kW-1)))
#for n_img in range(n):
for c in range(oC):
    for h in range(H-(kH-1)):
        for w in range(W-(kW-1)):
            out[:,:,h,w] = np.sum((input[:,:,h:h+kH,w:w+kW]*kernel[c,:,:,:,]))

print(out)