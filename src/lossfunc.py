import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label).to(device))
        self.register_buffer("fake_label", torch.tensor(target_fake_label).to(device))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

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
        # 修改成激活函数前
        for x in range(3):
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(4, 8): #(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 13): #(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 22): #(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 31):#(21, 30):
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
        self.weight = [0.1,0.1,1,1,1]

    def forward(self,x,y):
        x_vgg,y_vgg = self.vgg(x),self.vgg(y)
        loss = 0.0
        for iter,(x_fea,y_fea) in enumerate(zip(x_vgg,y_vgg)):
            loss += self.criterion(x_fea,y_fea.detach())*self.weight[iter]
        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_2x = torch.Tensor(1, 2, 3, 3)
        sobel_2y = torch.Tensor(1, 2, 3, 3)
        sobel_2x[:, 0:2, :, :] = sobel_x
        sobel_2y[:, 0:2, :, :] = sobel_y
        self.conv_hx = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.conv_hy = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.conv_hx.weight = torch.nn.Parameter(sobel_2x)
        self.conv_hy.weight = torch.nn.Parameter(sobel_2y)
        self.conv_hx = self.conv_hx.to(device)
        self.conv_hy = self.conv_hy.to(device)
    def forward(self, X, Y):
        X_hx = self.conv_hx(X)
        X_hy = self.conv_hy(Y)
        G_X = torch.abs(X_hx) + torch.abs(X_hy)
        # compute gradient of Y
        Y_hx = self.conv_hx(Y)
        self.conv_hx.train(False)
        Y_hy = self.conv_hy(Y)
        self.conv_hy.train(False)
        G_Y = torch.abs(Y_hx) + torch.abs(Y_hy)
        loss = F.mse_loss(G_X, G_Y, size_average=True)
        return loss



   
    
