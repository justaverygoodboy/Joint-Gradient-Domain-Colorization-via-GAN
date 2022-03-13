import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def Edge_x(im): #输入的是四维tensor(batch,channel,height,width)
    conv_op = nn.Conv2d(1,1,3,bias=False)
    sobel_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
    sobel_kernel = sobel_kernel.reshape((1,1,3,3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    edge_x = conv_op(Variable(im))
    edge_x = edge_x.squeeze()
    return edge_x
def Edge_y(im):
    conv_op = nn.Conv2d(1,1,3,bias=False)
    sobel_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float32)
    sobel_kernel = sobel_kernel.reshape((1,1,3,3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    edge_y = conv_op(Variable(im))
    edge_y = edge_y.squeeze()
    return edge_y

## 这里要注意是不是要把图像转成RGB的再求loss，因为如果按灰度来的话他们梯度应该是一样的，应该要使用结合颜色信息的RGB
# 导入GradientLoss然后传入两个值即可，这两个值就是图片再nn中的四维向量
## 这个是一阶梯度，sobel算子
class GradientLoss_S(nn.Module):
    def __init__(self):
        super(GradientLoss_S, self).__init__()
        sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_3x = torch.Tensor(1, 3, 3, 3)
        sobel_3y = torch.Tensor(1, 3, 3, 3)
        sobel_3x[:, 0:3, :, :] = sobel_x
        sobel_3y[:, 0:3, :, :] = sobel_y
        self.conv_hx = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_hy = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_hx.weight = torch.nn.Parameter(sobel_3x)
        self.conv_hy.weight = torch.nn.Parameter(sobel_3y)
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

   
    
