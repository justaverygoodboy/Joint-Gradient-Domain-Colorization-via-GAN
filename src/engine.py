import config
import torch
from tqdm import tqdm
from torch import nn
from lossfunc import PerceptualLoss
from lossfunc import GradientLoss

def train(train_loader, GAN_Model, netD, optG, optD, device, losses):
  batch = 0
  def wgan_loss(prediction, real_or_not):
    if real_or_not:
      return -torch.mean(prediction.float())
    else:
      return torch.mean(prediction.float())

  def gp_loss(y_pred, averaged_samples, gradient_penalty_weight):
    gradients = torch.autograd.grad(y_pred,averaged_samples,
                              grad_outputs=torch.ones(y_pred.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = (((gradients+1e-16).norm(2, dim=1) - 1) ** 2).mean() * gradient_penalty_weight
    return gradient_penalty

  ## 迭代读取数据，直接分离了L和AB通道
  for trainL, trainAB, _ in tqdm(iter(train_loader)):
      batch += 1  
      ########### add noise #################
      z = torch.rand((trainL.size(0),2,128,128),device=device)
      # trainL_3 = torch.tensor(np.tile(trainL.cpu(), [1,3,1,1]), device=device).float() #这里要不要把ab通道改成噪音
      trainL = torch.tensor(trainL, device=device).float()
      trainAB = torch.tensor(trainAB, device=device).float()
      trainL_3 = torch.cat([trainL,z],dim=1) # add noise to grayscale image for training
      ############ GAN MODEL ( Training Generator) ###################
      optG.zero_grad()
      predAB, discpred = GAN_Model(trainL, trainL_3) #得到预测的AB、类向量、D辨别结果
      D_G_z1 = discpred.mean().item() # GAN生成后进入D的判别结果
      ############ 先获得真实的图像和生成的图像 #############
      realLAB = torch.cat([trainL, trainAB], dim=1) #真实的图像
      predLAB = torch.cat([trainL, predAB], dim=1) #生成器得到的：预测的图像
      ############ Loss ##################################
      Loss_WL = wgan_loss(discpred, True) # WL是辨别器输出和真实的loss
      Loss_MSE = nn.MSELoss()(predAB, trainAB) #MSE是预测的AB和真实AB的L2
      Loss_Percp = PerceptualLoss()(predLAB,realLAB)
      Loss_Gradient = GradientLoss()(predAB,trainAB)
      #############
      Loss_G = Loss_WL*0.1 + Loss_MSE + Loss_Percp*0.003 + Loss_Gradient*0.01 #总loss
      Loss_G.backward()
      optG.step() # 使用生成网络的优化器优化
      losses['G_losses'].append(Loss_G.item())
      losses['EPOCH_G_losses'].append(Loss_G.item())
      ################################################################
      ############### Discriminator Training #########################
      for param in netD.parameters(): # 将D的设置为可BP
        param.requires_grad = True
      optD.zero_grad()
      discpred = netD(predLAB.detach()) #预测图像辨别结果
      D_G_z2 = discpred.mean().item() #均值
      discreal = netD(realLAB) #真实图像辨别结果
      D_x = discreal.mean().item() #均值
      weights = torch.randn((trainAB.size(0),1,1,1), device=device)          
      averaged_samples = (weights * trainAB ) + ((1 - weights) * predAB.detach())
      averaged_samples = torch.autograd.Variable(averaged_samples, requires_grad=True)
      avg_img = torch.cat([trainL, averaged_samples], dim=1)
      discavg = netD(avg_img)
      Loss_D_Fake = wgan_loss(discpred, False)
      Loss_D_Real = wgan_loss(discreal, True)
      Loss_D_avg = gp_loss(discavg, averaged_samples, config.GRADIENT_PENALTY_WEIGHT)
      Loss_D = Loss_D_Fake + Loss_D_Real + Loss_D_avg
      Loss_D.backward()
      optD.step()
      losses['D_losses'].append(Loss_D.item())
      losses['EPOCH_D_losses'].append(Loss_D.item())
      # Output training stats
      if batch % 10 == 0: #原本是100
        print('MSE: %.8f | Percp: %.8f | Loss_Grad:%.8f| Loss_D: %.8f | Loss_G: %.8f | D(x): %.8f | D(G(z)): %.8f | WGAN_F(G): %.8f | WGAN_F(D): %.8f | WGAN_R(D): %.8f | WGAN_A(D): %.8f | '
            % (Loss_MSE.item(),Loss_Percp.item(),Loss_Gradient.item(),Loss_D.item(), Loss_G.item(), D_x, D_G_z1, Loss_WL.item(), Loss_D_Fake.item(), Loss_D_Real.item(), Loss_D_avg.item()))

      