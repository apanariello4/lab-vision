import random
import numpy as np

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(1, 2)
input = np.random.rand(n, iC, H, W)


oH = int((H - kH)/s + 1)
oW = int((W - kW)/s + 1)

out = np.zeros((n,iC,oH,oW))

for h in range(oH):
    for w in range(oW):
        out[:,:,h,w] = input[:,:,h*s:h*s+kH,w*s:w*s+kW].max(axis=(2, 3))

print(out)