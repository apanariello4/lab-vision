import random
import numpy as np

n = random.randint(1, 3)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = np.random.rand(n, H, W).astype(np.float32)
template = np.random.rand(kH, kW).astype(np.float32)
#############################################

oH=H-(kH-1)
oW=W-(kW-1)

out = np.zeros((n, oH, oW))

def matching(input,template):
    for h in range(oH):
        for w in range(oW):
            this_template = np.expand_dims(template, 0)
            out[:,h,w] = np.sum(np.square(this_template - input[:,h:h+kH,w:w+kW]), axis=(-1,-2))
        return out

out = matching(input,template)
#print(out)