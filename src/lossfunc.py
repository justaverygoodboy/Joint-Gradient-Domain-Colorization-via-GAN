import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4)
    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)
        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)
        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)
        x = x.view(b, c, h, w)
        return x

class BCELoss(nn.Module):
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
        for x in range(3, 7): #(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12): #(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21): #(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):#(21, 30):
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
        return loss*0.2

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.sobel = SobelOperator(1e-4).cuda()

    def forward(self, pr, gt):
        gt_sobel = self.sobel(gt)
        pr_sobel = self.sobel(pr)
        grad_loss = F.l1_loss(gt_sobel, pr_sobel)
        return grad_loss
   
    
def dis_loss_hinge_fake(dis_fake):
    loss_fake = torch.mean(F.relu(1.+dis_fake))
    return loss_fake

def dis_loss_hinge_real(dis_real):
    loss_real = torch.mean(F.relu(1.-dis_real))
    return loss_real

def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss

def loss_hinge_dis(dis_fake,dis_real):
    loss_real = dis_loss_hinge_real(dis_real)
    loss_fake = dis_loss_hinge_fake(dis_fake)
    return loss_real+loss_fake

