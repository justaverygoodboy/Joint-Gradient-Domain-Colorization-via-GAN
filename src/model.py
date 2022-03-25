import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.layers import unetConv2
from unet.init_weights import init_weights
from unet.layers import UpsampleBLock
bias=True

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        # 这里之所以采用卷积是因为卷积中的权重是可学习的
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1)) # y = out*gamma + x 
        
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps(B,C,W,H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        # fx 得到B*N*C形状
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        # gx 得到B*C*N形状
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        # fx*gx^T bmm貌似李沐讲过
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # 得到 attention map，形状B*N*N
        attention = self.softmax(energy) # BX (N) X (N) 
        # hx，得到B*C*N形状
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        # hx和attention map相乘
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class UNet_3Plus(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]
        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.down_conv1 = nn.Conv2d(filters[0],filters[0],3,2,1)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.down_conv2 = nn.Conv2d(filters[1],filters[1],3,2,1)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.down_conv3 = nn.Conv2d(filters[2],filters[2],3,2,1)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.down_conv4 = nn.Conv2d(filters[3],filters[3],3,2,1)
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks # 64*5

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True) #kernel=8,stride=8 #尺寸除以8
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1) #64->64 #通道变换，尺寸不变
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_GeLU = nn.GeLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True) #尺寸除以4
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1) #128->64
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_GeLU = nn.GeLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True) #尺寸除以2
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1) #256->64
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_GeLU = nn.GeLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1) #512->64
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        # self.hd5_UT_hd4 = nn.ConvTranspose2d(1024,1024,kernel_size=4, stride=2, padding=1)  # 14*14
        self.hd5_UT_hd4 = UpsampleBLock(1024,2);
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1) #1024->64
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_GeLU = nn.GeLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU4d_1 = nn.GeLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_GeLU = nn.GeLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_GeLU = nn.GeLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_GeLU = nn.GeLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        # self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3 = UpsampleBLock(320,2)
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        # self.hd5_UT_hd3 = nn.ConvTranspose2d(1024,1024,4,4)  # 14*14
        self.hd5_UT_hd3 = UpsampleBLock(1024,4)
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_GeLU = nn.GeLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU3d_1 = nn.GeLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_GeLU = nn.GeLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_GeLU = nn.GeLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        # self.hd3_UT_hd2 = nn.ConvTranspose2d(320,320,kernel_size=4, stride=2, padding=1)  # 14*14
        self.hd3_UT_hd2 = UpsampleBLock(320,2)
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_GeLU = nn.GeLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        # self.hd4_UT_hd2 = nn.ConvTranspose2d(320,320,4,4)  # 14*14
        self.hd4_UT_hd2 = UpsampleBLock(320,4)
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        # self.hd5_UT_hd2 = nn.ConvTranspose2d(1024,1024,8,8)  # 14*14
        # self.hd5_UT_hd2 = UpsampleBLock(1024,8) # this is too many channels
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_GeLU = nn.GeLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU2d_1 = nn.GeLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_GeLU = nn.GeLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        # self.hd2_UT_hd1 = nn.ConvTranspose2d(320,320,kernel_size=4, stride=2, padding=1)  # 14*14
        self.hd2_UT_hd1 = UpsampleBLock(320,2)
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_GeLU = nn.GeLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        # self.hd3_UT_hd1 = nn.ConvTranspose2d(320,320,4,4)  # 14*14
        self.hd3_UT_hd1 = UpsampleBLock(320,4)
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_GeLU = nn.GeLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        # self.hd4_UT_hd1 = nn.ConvTranspose2d(320,320,8,8)  # 14*14
        self.hd4_UT_hd1 = UpsampleBLock(320,8)
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        # self.hd5_UT_hd1 = nn.ConvTranspose2d(1024,1024,16,16)  # 14*14
        # self.hd5_UT_hd1 = UpsampleBLock(1024,16) # There is too largek, using bilinear instead
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_GeLU = nn.GeLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU1d_1 = nn.GeLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

        self.attn1 = Self_Attn( 320, 'GeLU') 
        # self.attn2 = Self_Attn( 320,  'GeLU') 

    def forward(self, inputs): # b,3,128,128
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # b,64,128，128

        # h2 = self.maxpool1(h1) # b,64,64,64
        h2 = self.down_conv1(h1)
        h2 = self.conv2(h2)  # b,128,64,64

        # h3 = self.maxpool2(h2) # b,128,32,32
        h3 = self.down_conv2(h2)
        h3 = self.conv3(h3)  # b,256,32,32

        # h4 = self.maxpool3(h3) # b,256,16,16
        h4 = self.down_conv3(h3)
        h4 = self.conv4(h4)  # b,512,16,16

        # h5 = self.maxpool4(h4) # b,512,8,8
        h5 = self.down_conv4(h4)
        hd5 = self.conv5(h5)  # b,1024,8,8

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_GeLU(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))) #b,64,16,16
        h2_PT_hd4 = self.h2_PT_hd4_GeLU(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))) #b,64,16,16
        h3_PT_hd4 = self.h3_PT_hd4_GeLU(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))) #b,64,16,16
        h4_Cat_hd4 = self.h4_Cat_hd4_GeLU(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4))) #b,64,16,16
        hd5_UT_hd4 = self.hd5_UT_hd4_GeLU(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))) #b,64,16,16
        # hd4 = self.GeLU4d_1(self.bn4d_1(self.conv4d_1(
        #     torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))
        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))
        # b,320,16,16
        h1_PT_hd3 = self.h1_PT_hd3_GeLU(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))) #b,64,32,32
        h2_PT_hd3 = self.h2_PT_hd3_GeLU(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))) #b,64,32,32
        h3_Cat_hd3 = self.h3_Cat_hd3_GeLU(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3))) #b,64,32,32
        hd4_UT_hd3 = self.hd4_UT_hd3_GeLU(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))) #b,64,32,32
        hd5_UT_hd3 = self.hd5_UT_hd3_GeLU(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))) #b,64,32,32
        # hd3 = self.GeLU3d_1(self.bn3d_1(self.conv3d_1(
        #     torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) #b,320,32,32
        hd3 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))
        h1_PT_hd2 = self.h1_PT_hd2_GeLU(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))) #b,64,64,64
        h2_Cat_hd2 = self.h2_Cat_hd2_GeLU(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2))) #b,64,64,64
        hd3_UT_hd2 = self.hd3_UT_hd2_GeLU(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))) #b,64,64,64
        hd4_UT_hd2 = self.hd4_UT_hd2_GeLU(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))) #b,64,64,64
        hd5_UT_hd2 = self.hd5_UT_hd2_GeLU(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))) #b,64,64,64
        # hd2 = self.GeLU2d_1(self.bn2d_1(self.conv2d_1(
        #     torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) #b,320,64,64
        # hd2,p2 = self.attn2(hd2)
        hd2 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))
        h1_Cat_hd1 = self.h1_Cat_hd1_GeLU(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1))) #b,64,128,128
        hd2_UT_hd1 = self.hd2_UT_hd1_GeLU(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))) #b,64,128,128
        hd3_UT_hd1 = self.hd3_UT_hd1_GeLU(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))) #b,64,128,128
        hd4_UT_hd1 = self.hd4_UT_hd1_GeLU(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))) #b,64,128,128
        hd5_UT_hd1 = self.hd5_UT_hd1_GeLU(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))) #b,64,128,128
        # hd1 = self.GeLU1d_1(self.bn1d_1(self.conv1d_1(
        #     torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # b,320,128,128
        hd1 = self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))
        hd1,p1 = self.attn1(hd1)
        d1 = self.outconv1(hd1)  # b,2,128,128
        return F.sigmoid(d1)

class UNet_3Plus_AE(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_AE, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]
        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.down_conv1 = nn.Conv2d(filters[0],filters[0],3,2,1)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.down_conv2 = nn.Conv2d(filters[1],filters[1],3,2,1)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.down_conv3 = nn.Conv2d(filters[2],filters[2],3,2,1)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.down_conv4 = nn.Conv2d(filters[3],filters[3],3,2,1)
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks # 64*5

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True) #kernel=8,stride=8 #尺寸除以8
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1) #64->64 #通道变换，尺寸不变
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_GeLU = nn.GeLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True) #尺寸除以4
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1) #128->64
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_GeLU = nn.GeLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True) #尺寸除以2
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1) #256->64
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_GeLU = nn.GeLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1) #512->64
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = UpsampleBLock(1024,2);
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1) #1024->64
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_GeLU = nn.GeLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU4d_1 = nn.GeLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_GeLU = nn.GeLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_GeLU = nn.GeLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_GeLU = nn.GeLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = UpsampleBLock(320,2)
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = UpsampleBLock(1024,4)
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_GeLU = nn.GeLU(inplace=True)

        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU3d_1 = nn.GeLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_GeLU = nn.GeLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_GeLU = nn.GeLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = UpsampleBLock(320,2)
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_GeLU = nn.GeLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = UpsampleBLock(320,4)
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_GeLU = nn.GeLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU2d_1 = nn.GeLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_GeLU = nn.GeLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = UpsampleBLock(320,2)
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_GeLU = nn.GeLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = UpsampleBLock(320,4)
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_GeLU = nn.GeLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = UpsampleBLock(320,8)
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_GeLU = nn.GeLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_GeLU = nn.GeLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.GeLU1d_1 = nn.GeLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs): # b,3,128,128
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # b,64,128，128

        h2 = self.down_conv1(h1)
        h2 = self.conv2(h2)  # b,128,64,64

        h3 = self.down_conv2(h2)
        h3 = self.conv3(h3)  # b,256,32,32

        h4 = self.down_conv3(h3)
        h4 = self.conv4(h4)  # b,512,16,16

        h5 = self.down_conv4(h4)
        hd5 = self.conv5(h5)  # b,1024,8,8

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_GeLU(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))) #b,64,16,16
        h2_PT_hd4 = self.h2_PT_hd4_GeLU(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))) #b,64,16,16
        h3_PT_hd4 = self.h3_PT_hd4_GeLU(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))) #b,64,16,16
        h4_Cat_hd4 = self.h4_Cat_hd4_GeLU(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4))) #b,64,16,16
        hd5_UT_hd4 = self.hd5_UT_hd4_GeLU(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))) #b,64,16,16
        hd4 = self.GeLU4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))
        # b,320,16,16
        h1_PT_hd3 = self.h1_PT_hd3_GeLU(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))) #b,64,32,32
        h2_PT_hd3 = self.h2_PT_hd3_GeLU(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))) #b,64,32,32
        h3_Cat_hd3 = self.h3_Cat_hd3_GeLU(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3))) #b,64,32,32
        hd4_UT_hd3 = self.hd4_UT_hd3_GeLU(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))) #b,64,32,32
        hd5_UT_hd3 = self.hd5_UT_hd3_GeLU(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))) #b,64,32,32
        hd3 = self.GeLU3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) #b,320,32,32
       
        h1_PT_hd2 = self.h1_PT_hd2_GeLU(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))) #b,64,64,64
        h2_Cat_hd2 = self.h2_Cat_hd2_GeLU(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2))) #b,64,64,64
        hd3_UT_hd2 = self.hd3_UT_hd2_GeLU(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))) #b,64,64,64
        hd4_UT_hd2 = self.hd4_UT_hd2_GeLU(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))) #b,64,64,64
        hd5_UT_hd2 = self.hd5_UT_hd2_GeLU(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))) #b,64,64,64
        hd2 = self.GeLU2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) #b,320,64,64
        # hd2,p2 = self.attn2(hd2)
        
        h1_Cat_hd1 = self.h1_Cat_hd1_GeLU(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1))) #b,64,128,128
        hd2_UT_hd1 = self.hd2_UT_hd1_GeLU(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))) #b,64,128,128
        hd3_UT_hd1 = self.hd3_UT_hd1_GeLU(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))) #b,64,128,128
        hd4_UT_hd1 = self.hd4_UT_hd1_GeLU(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))) #b,64,128,128
        hd5_UT_hd1 = self.hd5_UT_hd1_GeLU(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))) #b,64,128,128
        hd1 = self.GeLU1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # b,320,128,128
        # hd1,p1 = self.attn1(hd1)
        d1 = self.outconv1(hd1)  # b,2,128,128
        return F.sigmoid(d1)

# net D
class discriminator_model(nn.Module):
  def __init__(self):
    super(discriminator_model, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 4, 2, 1) # 64,64,64
    self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # 128,32,32
    self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # 256,16,16
    self.conv4 = nn.Conv2d(256, 512, 4, 2, 1) # 512,8,8
    self.conv5 = nn.Conv2d(512, 1, 4) # 1,5,5 
    self.leaky_GeLU = nn.LeakyGeLU(0.1)
    self.attn1 = Self_Attn(256,'GeLU')
    self.attn2 = Self_Attn(512,'GeLU')

  def forward(self,input):
    net = self.conv1(input)              
    net = self.leaky_GeLU(net)         
    net = self.conv2(net)              
    net = self.leaky_GeLU(net)         
    net = self.conv3(net)              
    net = self.leaky_GeLU(net)          
    net,p1 = self.attn1(net)
    net = self.conv4(net)              
    net = self.leaky_GeLU(net)
    net,p2 = self.attn2(net)          
    net = self.conv5(net)               
    return net.squeeze()


class GAN(nn.Module):
  def __init__(self, netG, netD):
    super(GAN, self).__init__()
    self.netG = netG
    self.netD = netD

  def forward(self, trainL, trainL_3):
    for param in self.netD.parameters(): 
      param.requires_grad= False
    predAB = self.netG(trainL_3) 
    predLAB = torch.cat([trainL, predAB], dim=1) 
    discpred = self.netD(predLAB) 
    return predAB, discpred 

