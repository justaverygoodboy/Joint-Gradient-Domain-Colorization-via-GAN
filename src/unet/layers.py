import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_weights import init_weights


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        # if is_batchnorm:
        #     for i in range(1, n + 1): #两层 Conv-BN-ReLU
        #         conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), # (w-kz)+2p+1，这里尺寸不变
        #                              nn.BatchNorm2d(out_size),
        #                              nn.ReLU(inplace=True), )
        #         setattr(self, 'conv%d' % i, conv)
        #         in_size = out_size

        # else:
        #     for i in range(1, n + 1):
        #         conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
        #                              nn.ReLU(inplace=True), )
        #         setattr(self, 'conv%d' % i, conv)
        #         in_size = out_size
        conv = nn.Sequential(
            nn.Conv2d(in_size,out_size,ks,s,p),
            nn.BatchNorm2d(out_size),
            nn.Conv2d(out_size,out_size,ks,s,p),
            nn.GELU(inplace=True)
            )
        setattr(self,'conv',conv)
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        # for i in range(1, self.n + 1):
        #     conv = getattr(self, 'conv%d' % i)
        #     x = conv(x)
        conv = getattr(self,'conv')
        x = conv(x)
        return x


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
