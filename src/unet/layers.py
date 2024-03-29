import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_weights import init_weights

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        for m in self.children():
            init_weights(m, init_type='kaiming')
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
 

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        conv = nn.Sequential(
            nn.Conv2d(in_size,out_size,ks,s,p),
            nn.GELU(), #加入一层gelu
            nn.Conv2d(out_size,out_size,ks,s,p),
            nn.GELU()
            )
        setattr(self,'conv',conv)
        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        conv = getattr(self,'conv')
        x = conv(x)
        return x


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels*up_scale**2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        # x = self.prelu(x)
        return x
