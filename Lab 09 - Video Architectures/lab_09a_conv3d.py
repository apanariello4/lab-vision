import random
import torch
from math import floor

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, T, H, W)
kernel = torch.rand(oC, iC, kT, kH, kW)
bias = torch.rand(oC)


def conv3d(input, kernel, bias, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    n, iC, T, H, W = input.shape
    oC, iC, kT, kH, kW = kernel.shape

    oT = floor(((T+2*padding[0]-dilation[0]*(kT-1)-1)/stride[0])+1)
    oH = floor(((H+2*padding[1]-dilation[1]*(kH-1)-1)/stride[1])+1)
    oW = floor(((W+2*padding[2]-dilation[2]*(kW-1)-1)/stride[2])+1)

    output = torch.zeros((n, oC, oT, oH, oW))

    for t in range(oT):
        for h in range(oH):
            for w in range(oW):
                output[:, :, t, h, w] = torch.sum(
                    torch.unsqueeze(kernel[:, :, :, :, :], 0) * torch.unsqueeze(input[:, :, t:t+kT, h:h+kH, w:w+kW], 1), (-1, -2, -3, -4)) + bias

    return output


out = conv3d(input, kernel, bias)
