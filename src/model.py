import torch
import torch.nn as nn
import torchvision
import numpy as np
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

class SAResGenerator(nn.Module):
    """Generator."""
    def __init__(self):
        super(SAResGenerator, self).__init__()

        # 加个resnet152做主干
        self.ResNet = torchvision.models.resnet152(pretrained=True)
        self.ResNet = nn.Sequential(*list(self.ResNet.children())[:-2]) # b,2048,4,4
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []
        last = []
        layer1.append(nn.ConvTranspose2d(2048, 1024, 4)) #2048->1024 1024,16,7,7
        layer1.append(nn.BatchNorm2d(1024))
        layer1.append(nn.ReLU())
        layer2.append(nn.ConvTranspose2d(1024, 512, 4, 2, 1)) #b,512,14,14
        layer2.append(nn.BatchNorm2d(512))
        layer2.append(nn.ReLU())
        layer3.append(nn.ConvTranspose2d(512, 256, 4, 2, 1)) #b,256,28,28
        layer3.append(nn.BatchNorm2d(256))
        layer3.append(nn.ReLU())
        layer4.append(nn.ConvTranspose2d(256, 128, 4, 2, 1)) #b,128,56,56
        layer4.append(nn.BatchNorm2d(128))
        layer4.append(nn.ReLU())
        layer5.append(nn.ConvTranspose2d(128,64,4,2,1)) #b,64,h,w
        layer5.append(nn.BatchNorm2d(64))
        layer5.append(nn.ReLU())
        layer6.append(nn.ConvTranspose2d(64,32,9)) #b,32,h,w
        layer6.append(nn.BatchNorm2d(32))
        layer6.append(nn.ReLU())
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)
        self.l6 = nn.Sequential(*layer6)
        last.append(nn.ConvTranspose2d(32, 2, 9)) #32,2,128,128
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)
        self.attn1 = Self_Attn( 128, 'relu') #这里应该改256
        self.attn2 = Self_Attn( 64,  'relu') #这里应该改128

    def forward(self, img):
        # z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.ResNet(img)
        out=self.l1(out)
        out=self.l2(out)
        out=self.l3(out)
        out=self.l4(out)
        out,p1 = self.attn1(out)
        out=self.l5(out)
        out,p2 = self.attn2(out)
        out=self.l6(out)
        out=self.last(out)
        return out

# net D
class discriminator_model(nn.Module):
  def __init__(self):
    super(discriminator_model, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=(4,4),padding=1,stride=(2,2),bias=bias) # 64, 112, 112
    self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), padding=1, stride=(2,2), bias=bias) # 128, 56, 56
    self.conv3 = nn.Conv2d(128,256, kernel_size=(4,4), padding=1, stride=(2,2), bias=bias) # 256, 28, 28, 2
    self.conv4 = nn.Conv2d(256,512, kernel_size=(4,4), padding=3, stride=(1,1), bias=bias) # 512, 28, 28
    self.conv5 = nn.Conv2d(512,1, kernel_size=(4,4), padding=3, stride=(1,1), bias=bias) # 1, 
    self.leaky_relu = nn.LeakyReLU(0.3)

  def forward(self,input):
    net = self.conv1(input)               #[-1, 64, 112, 112]
    net = self.leaky_relu(net)          #[-1, 64, 112, 112]    
    net = self.conv2(net)               #[-1, 128, 56, 56] 
    net = self.leaky_relu(net)          #[-1, 128, 56, 56] 
    net = self.conv3(net)               #[-1, 256, 28, 28]
    net = self.leaky_relu(net)          #[-1, 256, 28, 28]   
    net = self.conv4(net)               #[-1, 512, 27, 27]
    net = self.leaky_relu(net)          #[-1, 512, 27, 27]
    net = self.conv5(net)               #[-1, 1, 26, 26]
    return net

# net G
# class colorization_model(nn.Module):
#   def __init__(self):
#     super(colorization_model, self).__init__()
#     self.VGG_model = torchvision.models.vgg16(pretrained=True)
#     self.VGG_model = nn.Sequential(*list(self.VGG_model.features.children())[:-8]) #[None, 512, 28, 28] 这个貌似是对模型进行修改，除去了顶层，#
#     self.VGG_model = self.VGG_model #double是指数据类型，从float变double
#     self.relu = nn.ReLU()
#     self.lrelu = nn.LeakyReLU(0.3)
#     self.global_features_conv1 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(2,2), bias=bias) #[None, 512, 14, 14]
#     self.global_features_bn1 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
#     self.global_features_conv2 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 14, 14]
#     self.global_features_bn2 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
#     self.global_features_conv3 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(2,2), bias=bias) #[None, 512, 7, 7]
#     self.global_features_bn3 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
#     self.global_features_conv4 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 7, 7]
#     self.global_features_bn4 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)

#     self.global_features2_flatten = nn.Flatten()
#     self.global_features2_dense1 = nn.Linear(512*7*7,1024)
#     self.midlevel_conv1 = nn.Conv2d(512,512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 28, 28]
#     self.global_features2_dense2 = nn.Linear(1024,512)
#     self.midlevel_bn1 = nn.BatchNorm2d(512, eps=0.001,momentum=0.99)
#     self.global_features2_dense3 = nn.Linear(512,256)
#     self.midlevel_conv2 = nn.Conv2d(512,256, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
#     self.midlevel_bn2 = nn.BatchNorm2d(256,eps=0.001,momentum=0.99)
#      #[None, 256, 28, 28]
#     # self.midlevel_bn2 = nn.BatchNorm2d(256)#,,eps=0.001,momentum=0.99)
#     self.global_featuresClass_flatten = nn.Flatten()
#     self.global_featuresClass_dense1 = nn.Linear(512*7*7, 4096)
#     self.global_featuresClass_dense2 = nn.Linear(4096, 4096)
#     self.global_featuresClass_dense3 = nn.Linear(4096, 1000)
#     self.softmax = nn.Softmax()

#     self.outputmodel_conv1 = nn.Conv2d(512, 256, kernel_size=(1,1), padding=0, stride=(1,1),  bias=bias) 
#     self.outputmodel_conv2 = nn.Conv2d(256, 128, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
#     self.outputmodel_conv3 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
#     self.outputmodel_conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
#     self.outputmodel_conv5 = nn.Conv2d(64, 32, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
#     self.outputmodel_conv6 = nn.Conv2d(32, 2, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
#     self.outputmodel_upsample = nn.Upsample(scale_factor=(2,2))
#     self.outputmodel_bn1 = nn.BatchNorm2d(128)
#     self.outputmodel_bn2 = nn.BatchNorm2d(64)
#     self.sigmoid = nn.Sigmoid()
#     self.tanh = nn.Tanh()

  # def forward(self,input_img):
  #   # VGG Without Top Layers 黄色部分
  #   vgg_out = self.VGG_model(torch.tensor(input_img))
  #   #Global Features 红色部分
  #   global_features = self.relu(self.global_features_conv1(vgg_out))  #[None, 512, 14, 14]
  #   global_features = self.global_features_bn1(global_features) #[None, 512, 14, 14]
  #   global_features = self.relu(self.global_features_conv2(global_features)) #[None, 512, 14, 14]
  #   global_features = self.global_features_bn2(global_features) #[None, 512, 14, 14]
  #   global_features = self.relu(self.global_features_conv3(global_features)) #[None, 512, 7, 7]
  #   global_features = self.global_features_bn3(global_features)  #[None, 512, 7, 7]


  #   # Fusion Colorization 蓝色部分
  #   outputmodel = self.relu(self.outputmodel_conv1(global_features)) # None, 256, 28, 28
  #   outputmodel = self.outputmodel_upsample(outputmodel) # None, 128, 56, 56
  #   outputmodel = self.relu(self.outputmodel_conv2(outputmodel)) # None, 128, 28, 28
  #   outputmodel = self.outputmodel_upsample(outputmodel) # None, 128, 56, 56
  #   outputmodel = self.outputmodel_bn1(outputmodel) # None, 128, 56, 56
  #   outputmodel = self.relu(self.outputmodel_conv3(outputmodel)) # None, 64, 56, 56
  #   outputmodel = self.outputmodel_upsample(outputmodel) # None, 128, 56, 56
  #   outputmodel = self.relu(self.outputmodel_conv4(outputmodel)) # None, 64, 56, 56 
  #   outputmodel = self.outputmodel_upsample(outputmodel) # None, 64, 112, 112
  #   outputmodel = self.outputmodel_bn2(outputmodel) # None, 64, 112, 112
  #   outputmodel = self.relu(self.outputmodel_conv5(outputmodel)) # None, 32, 112, 112
  #   outputmodel = self.outputmodel_upsample(outputmodel) # None, 128, 56, 56
  #   outputmodel = self.sigmoid(self.outputmodel_conv6(outputmodel)) # None, 2, 112, 112
  #   return outputmodel #返回 预测的AB 和 类向量

class GAN(nn.Module):
  def __init__(self, netG, netD):
    super(GAN, self).__init__()
    self.netG = netG
    self.netD = netD

  def forward(self, trainL, trainL_3):
    # trainL用于拼接AB，trainL_3输入生成网络得到AB和类向量
    for param in self.netD.parameters(): #GAN的传播先将D设置成不需要BP
      param.requires_grad= False
    predAB = self.netG(trainL_3) #通过G得到预测的AB值、类向量
    predLAB = torch.cat([trainL, predAB], dim=1) #LAB拼起来得到最终结果
    discpred = self.netD(predLAB) # 将最终结果输入到D中，得到辨别结果
    return predAB, discpred #返回预测的AB值，类向量、D判别结果

