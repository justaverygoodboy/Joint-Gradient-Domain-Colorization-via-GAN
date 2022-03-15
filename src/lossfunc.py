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
        self.weights = [1.0,1.0,1.0,1.0,1.0]
        self.downsample = nn.AvgPool2d(2,2,count_include_pad=False)
    def forward(self,x,y):
        x_vgg,y_vgg = self.vgg(x),self.vgg(y)
        loss = 0.0
        for iter,(x_fea,y_fea) in enumerate(zip(x_vgg,y_vgg)):
            loss += self.criterion(x_fea,y_fea.detach())
        return loss


   
    
