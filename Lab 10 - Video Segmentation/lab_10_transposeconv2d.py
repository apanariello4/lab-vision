import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
s = random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)


def transpose(input, kernel, stride, padding=(0, 0), dilation=(1, 1)):
    n, iC, H, W = input.shape
    iC, oC, kH, kW = kernel.shape

    oH = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + 1
    oW = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + 1

    output = torch.zeros(n, oC, oH, oW)

    kernel = kernel.transpose(0, 1)
    kernel = kernel.unsqueeze(0)

    for h in range(H):
        for w in range(W):
            this_input = input[:, :, h, w].unsqueeze(
                1).unsqueeze(3).unsqueeze(4)
            output[:, :, h*stride[0]:h*stride[0]+kH, w*stride[1]:w*stride[1]+kW] += torch.sum(this_input *
                                                                                              kernel, dim=2)
    return output


out = transpose(input, kernel, (s, s))
