from turtle import forward
from xml.parsers.expat import model
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vgg19_out(nn.Module):
    def __init__(self):
        super(Vgg19_out,self).__init__()
        vgg = models.vgg19(pretrained=True).to(device)
        vgg.eval()
        vgg_pretrained_features = vgg.features
        self.requires_grad = False
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(4, 9): #(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14): #(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23): #(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):#(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss,self).__init__()
        self.vgg = Vgg19_out().to(device)
        self.criterion = nn.MSELoss()

    def forward(self,x,y):
        x_vgg,y_vgg = self.vgg(x),self.vgg(y)
        loss = 0.0
        for iter,(x_fea,y_fea) in enumerate(zip(x_vgg,y_vgg)):
            loss += self.criterion(x_fea,y_fea.detach())
        return loss

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient

def GradientLoss(x,y):
    gradient_model = Gradient_Net().to(device)
    gx = gradient_model(x)
    gy = gradient_model(y)
    criterion = nn.MSELoss()
    loss = criterion(gx,gy)
    return loss


   
    
