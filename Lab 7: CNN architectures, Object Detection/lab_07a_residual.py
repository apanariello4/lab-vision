import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        """

        :param inplanes: number of channels of input
        :param planes: number of output channels of conv1 and conv2
        :param stride: stride of conv1
        """
        super().__init__()
        self.inplanes, self.planes, self.stride = inplanes, planes, stride

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes)
        )

        self.batchnorm = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        if (self.inplanes != self.planes) or (self.stride > 1):
            residual = self.shortcut(x)

        x = F.relu(self.batchnorm(self.conv1(x)))
        return F.relu(self.batchnorm(self.conv2(x)) + residual)
