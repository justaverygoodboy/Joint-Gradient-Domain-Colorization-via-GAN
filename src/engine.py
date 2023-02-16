import config
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from lossfunc import PerceptualLoss
from lossfunc import GradientLoss
# from lossfunc import GANLoss
from lossfunc import SobelOperator

import numpy as np
import utils

def train(train_loader, GAN_Model, netDimg, netDgrad, optG, optDimg, optDgrad, device, losses): #这里要传入netDimg还要netDgrd
  batch = 0

  def wgan_loss(prediction, real_or_not):
    if real_or_not:
      return -torch.mean(prediction.float())
    else:
      return torch.mean(prediction.float())

  def hinge_loss(prediction, real_or_not):
    if real_or_not:
      return nn.ReLU()(1.0-prediction).mean()
    else:
      return nn.ReLU()(1.0+prediction).mean()

  # bce loss
  # adversarial_criterion = GANLoss()
  
  sobel = SobelOperator().cuda()
  noise = False

  for trainL, trainAB, _ in tqdm(iter(train_loader)):
      batch += 1  
      trainL = torch.tensor(trainL, device=device).float()
      trainAB = torch.tensor(trainAB, device=device).float()
      realLAB = torch.cat([trainL, trainAB], dim=1)
      if noise:
        z = torch.randn((trainL.size(0),2,config.IMAGE_SIZE,config.IMAGE_SIZE),device=device)
        trainL_3 = torch.cat([trainL,z],dim=1) # add noise to grayscale image for training
      else:
        trainL_3 = torch.tensor(np.tile(trainL.cpu(), [1,3,1,1]), device=device).float() #复制L为三个通道
      # 梯度域
      realSobel = sobel(realLAB)
      # utils.show_single_img(realSobel,3,sobel)

      optG.zero_grad()
      predAB,predLAB,predSobel,discpred_img,discpred_sobel = GAN_Model(trainL, trainL_3) 
      
      ############ G Loss ##################################
      # utils.show_single_channel(F.sigmoid(discpred.detach()),f'unet_gen{batch}')
      # Loss_adver = loss_hinge_gen(discpred)*0.001
      # Loss_adver = (adversarial_criterion(discpred_img,True)+adversarial_criterion(discpred_sobel,True))*0.01
      Loss_adver = (-discpred_img.mean()-discpred_sobel.mean())*0.1

      # Loss_WL = wgan_loss(discpred, True)*0.0001
      Loss_Pix = nn.L1Loss()(predAB, trainAB)
      Loss_Percp = PerceptualLoss()(predLAB,realLAB)*0.1
      Loss_Gradient = GradientLoss()(predLAB,realLAB)*10

      Loss_G = Loss_Pix+Loss_Gradient+Loss_adver+Loss_Percp
      # Loss_G = Loss_Pix
      Loss_G.backward()
      optG.step() 
      
      losses['G_losses'].append(Loss_G.item())
      losses['EPOCH_G_losses'].append(Loss_G.item())

      ############### Discriminator Training #########################
      # Dimg
      for param in netDimg.parameters(): # 将D的设置为可BP
        param.requires_grad = True
      optDimg.zero_grad()
      discpred_img = netDimg(predLAB.detach())
      discreal_img = netDimg(realLAB)
      # utils.show_single_channel(F.sigmoid(discpred.detach()),f'unet_real{batch}')
      # Loss_D_img_Fake = adversarial_criterion(discpred_img,False)
      # Loss_D_img_Real = adversarial_criterion(discreal_img,True)
      Loss_D_img_Fake = hinge_loss(discpred_img,False)
      Loss_D_img_Real = hinge_loss(discreal_img,True)
      # Loss_D_img_Fake = wgan_loss(discpred_img,False)
      # Loss_D_img_Real = wgan_loss(discreal_img,True)
      Loss_D_img = Loss_D_img_Fake+Loss_D_img_Real
      D_x_img = discreal_img.mean().item()
      D_G_img = discpred_img.mean().item() #生成img的判别值
      Loss_D_img.backward()
      optDimg.step()

      # Dsobel
      for param in netDgrad.parameters(): # 将D的设置为可BP
        param.requires_grad = True
      optDgrad.zero_grad()
      discpred_sobel = netDgrad(predSobel.detach())
      discreal_sobel = netDgrad(realSobel)
      # Loss_D_sobel_Fake = adversarial_criterion(discpred_sobel,False)
      # Loss_D_sobel_Real = adversarial_criterion(discreal_sobel,True)
      Loss_D_sobel_Fake = hinge_loss(discpred_sobel,False)
      Loss_D_sobel_Real = hinge_loss(discreal_sobel,True)
      # Loss_D_sobel_Fake = wgan_loss(discpred_sobel,False)
      # Loss_D_sobel_Real = wgan_loss(discreal_sobel,True)
      Loss_D_sobel = Loss_D_sobel_Fake+Loss_D_sobel_Real
      D_x_sobel = discreal_sobel.mean().item()
      D_G_sobel = discpred_sobel.mean().item() #生成sobel的判别值
      Loss_D_sobel.backward()
      optDgrad.step()

      # losses['D_losses'].append(Loss_D.item())
      # losses['EPOCH_D_losses'].append(Loss_D.item())

      # Output training stats
      if batch % 10 == 0: #原本是100
        print('L2: %.8f | Loss_Perc:%.8f |Loss_Grad:%.8f |Loss_adver:%.8f | Loss_D_img: %.8f |Loss_D_sobel: %.8f | Loss_G: %.8f | Dimg(x): %.8f | Dimg(G(z)): %.8f |Dsobel(x): %.8f | Dsobel(G(z)): %.8f|'
            % (Loss_Pix.item(),Loss_Percp.item(),Loss_Gradient.item(),Loss_adver.item(),Loss_D_img.item(),Loss_D_sobel.item(),Loss_G.item(), D_x_img, D_G_img, D_x_sobel, D_G_sobel))
      # if batch % 10 == 0: #原本是100
      #   print('L2: %.8f |Loss_Grad:%.8f| Loss_G: %.8f|'
      #       % (Loss_Pix.item(),Loss_Gradient.item(),Loss_G.item()))

      
      