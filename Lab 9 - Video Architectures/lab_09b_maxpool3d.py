import random
import torch
from math import floor

n = random.randint(2, 6)
iC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 5)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand(n, iC, T, H, W)


def maxpool3d(input, kernel, stride):
    n, iC, T, H, W = input.shape
    kT, kH, kW = kernel
    s = stride

    oT = floor((T - kT)/s + 1)
    oH = floor((H - kH)/s + 1)
    oW = floor((W - kW)/s + 1)

    output = torch.zeros((n, iC, oT, oH, oW))
    for t in range(oT):
        for h in range(oH):
            for w in range(oW):
                output[:, :, t, h, w] = torch.max(
                    torch.max(torch.max(input[:, :, t*s:t*s+kT, h*s:h*s+kH, w*s:w*s+kW], -3)[0], -2)[0], -1)[0]

    return output


out = maxpool3d(input, (kT, kH, kW), s)
